"""
fake_aruco_publisher
====================

시뮬레이션용 가상 ArUco 마커 발행기.

실제 카메라와 마커 없이도 "카메라가 마커를 본다"는 상황을 재현한다.
마커를 `base_link` 기준 고정 위치에 놓아두고, 매 주기마다 현재 `camera_link` 의
위치에서 **실제로 보이는지**(FOV / 거리 / 마커가 카메라를 향하는지)를 판정해
보이는 것만 카메라 좌표계로 발행한다.

`ros2 topic pub` 으로 좌표를 직접 주입하는 것과의 차이
-----------------------------------------------------
토픽 주입은 좌표가 항상 주어지므로 "팔이 그 자세에서 마커를 실제로 볼 수
있는가"를 검증하지 못한다. 이 노드는 카메라 시야를 따지므로:

  * Camera-Ready 자세가 작업 영역을 제대로 보는지 즉시 확인된다
  * 팔이 움직이면 마커가 보였다 안 보였다 하는 것까지 재현된다
  * RViz 에 물체와 카메라 시야가 그려져 눈으로 판단할 수 있다

[주의] 시야 판정은 `view_axis` 파라미터가 가리키는 `camera_link` 축을 렌즈가
바라보는 방향으로 본다. 이 프로젝트의 `camera_link` 는 ROS 카메라 바디 프레임
규약(REP-103, +X 전방)이라 기본값이 `x` 다. 마운트가 다르면 파라미터로 바꾼다.

[주의] 물체를 MoveIt 플래닝 씬의 충돌 객체로 넣지 않는다. 넣으면 그리퍼가
물체를 향해 내려가는 것 자체가 충돌로 판정돼 파지가 불가능해진다.
여기서는 눈으로 보기 위한 시각화 마커만 발행한다.
"""

import math

import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker

from piper_pnp.geometry import (
    quat_from_rpy,
    quat_normalize,
    quat_rotate_vector,
    transform_pose,
)

try:
    from ros2_aruco_interfaces.msg import ArucoMarkers
except ImportError:  # pragma: no cover
    ArucoMarkers = None

_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


class FakeArucoPublisher(Node):
    """가상 마커를 카메라 시야 판정과 함께 발행한다."""

    def __init__(self):
        super().__init__('fake_aruco_publisher')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_link')

        # 마커는 base_frame 기준 고정 위치에 놓인다. rpy 가 0 이면 마커 평면이
        # 수평이고 법선이 위(+Z)를 향한다 = 책상에 평평히 놓인 상태.
        self.declare_parameter('pick_marker_id', 0)
        self.declare_parameter('pick_marker_xyz', [0.30, 0.10, 0.05])
        self.declare_parameter('pick_marker_rpy', [0.0, 0.0, 0.0])
        self.declare_parameter('place_marker_id', 1)
        self.declare_parameter('place_marker_xyz', [0.30, -0.15, 0.05])
        self.declare_parameter('place_marker_rpy', [0.0, 0.0, 0.0])

        # 물체 크기 [x, y, z]. 마커는 이 상자의 윗면 중앙에 붙어 있다고 본다.
        self.declare_parameter('object_size', [0.04, 0.04, 0.05])
        self.declare_parameter('marker_size', 0.05)

        # D435 컬러 카메라 시야각 기본값
        self.declare_parameter('hfov_deg', 69.4)
        self.declare_parameter('vfov_deg', 42.5)
        self.declare_parameter('min_range', 0.10)
        self.declare_parameter('max_range', 2.00)
        # 마커 법선과 시선이 이보다 더 벌어지면 검출 실패로 본다
        self.declare_parameter('max_view_angle_deg', 75.0)
        # camera_link 는 REP-103 바디 프레임(+X 전방)이라 렌즈축은 +X.
        self.declare_parameter('view_axis', 'x')

        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('markers_topic', '/aruco_markers')
        self.declare_parameter('poses_topic', '/aruco_poses')
        self.declare_parameter('viz_topic', '/pnp_sim_objects')

        g = self.get_parameter
        self.base_frame = g('base_frame').value
        self.camera_frame = g('camera_frame').value
        self.object_size = list(g('object_size').value)
        self.marker_size = g('marker_size').value
        self.hfov = math.radians(g('hfov_deg').value)
        self.vfov = math.radians(g('vfov_deg').value)
        self.min_range = g('min_range').value
        self.max_range = g('max_range').value
        self.max_view_angle = math.radians(g('max_view_angle_deg').value)

        axis = str(g('view_axis').value).lower()
        if axis not in _AXIS_INDEX:
            raise ValueError(f"view_axis 는 x/y/z 중 하나여야 합니다 (받은 값: {axis})")
        self.view_axis = _AXIS_INDEX[axis]
        # 시야각을 재는 나머지 두 축 (가로, 세로)
        self.h_axis, self.v_axis = [i for i in (0, 1, 2) if i != self.view_axis]

        self._markers = []
        for prefix in ('pick', 'place'):
            xyz = list(g(f'{prefix}_marker_xyz').value)
            rpy = list(g(f'{prefix}_marker_rpy').value)
            self._markers.append({
                'name': prefix,
                'id': int(g(f'{prefix}_marker_id').value),
                'position': tuple(float(v) for v in xyz),
                'orientation': quat_from_rpy(*(float(v) for v in rpy)),
            })

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._pose_pub = self.create_publisher(PoseArray, g('poses_topic').value, 10)
        self._viz_pub = self.create_publisher(Marker, g('viz_topic').value, 10)
        self._marker_pub = None
        if ArucoMarkers is not None:
            self._marker_pub = self.create_publisher(
                ArucoMarkers, g('markers_topic').value, 10)
        else:
            self.get_logger().warn(
                'ros2_aruco_interfaces 가 없어 PoseArray 만 발행합니다 '
                '(두 마커가 모두 보일 때만).')

        self._last_visible = None
        self.create_timer(1.0 / g('publish_rate').value, self._tick)

        self.get_logger().info(
            f'가상 마커 {len(self._markers)}개 배치 완료 ({self.base_frame} 기준). '
            f'카메라={self.camera_frame}, 시야축=+{axis.upper()}, '
            f'FOV={g("hfov_deg").value:.1f}x{g("vfov_deg").value:.1f}deg')
        for m in self._markers:
            p = m['position']
            self.get_logger().info(
                f"  {m['name']:5s} id={m['id']} at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

    # ------------------------------------------------------------------
    def _tick(self):
        transform = self._lookup_base_to_camera()

        # 마커별 판정은 틱당 한 번만 하고 시각화와 발행이 그 결과를 함께 쓴다.
        results = [
            (marker,
             self._project_to_camera(marker, transform) if transform else None)
            for marker in self._markers
        ]
        self._publish_visualization(results)
        if transform is None:
            return

        visible = [(marker, seen) for marker, seen in results if seen is not None]
        names = tuple(m['name'] for m, _ in visible)
        if names != self._last_visible:
            self._last_visible = names
            self.get_logger().info(f'보이는 마커: {list(names) or "없음"}')

        if not visible:
            return

        stamp = self.get_clock().now().to_msg()

        if self._marker_pub is not None:
            msg = ArucoMarkers()
            msg.header.stamp = stamp
            msg.header.frame_id = self.camera_frame
            for marker, (position, orientation) in visible:
                msg.marker_ids.append(marker['id'])
                msg.poses.append(_make_pose(position, orientation))
            self._marker_pub.publish(msg)

        # PoseArray 에는 ID 가 없어 수신 측이 인덱스로 pick/place 를 가른다.
        # 순서가 어긋나면 안 되므로 둘 다 보일 때만 [pick, place] 순으로 낸다.
        if len(visible) == len(self._markers):
            poses = PoseArray()
            poses.header.stamp = stamp
            poses.header.frame_id = self.camera_frame
            for marker, (position, orientation) in visible:
                poses.poses.append(_make_pose(position, orientation))
            self._pose_pub.publish(poses)

    # ------------------------------------------------------------------
    def _lookup_base_to_camera(self):
        """base_frame 좌표를 camera_frame 으로 옮기는 변환."""
        try:
            tf = self._tf_buffer.lookup_transform(
                self.camera_frame, self.base_frame, Time())
        except TransformException as exc:
            self.get_logger().warn(
                f'{self.base_frame} -> {self.camera_frame} TF 대기 중: {exc}',
                throttle_duration_sec=5.0)
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        return (t.x, t.y, t.z), (r.x, r.y, r.z, r.w)

    def _project_to_camera(self, marker, transform):
        """
        마커를 카메라 좌표계로 옮기고 실제로 보이는지 판정한다.
        보이면 (position, orientation), 아니면 None.
        """
        translation, rotation = transform
        position, orientation = transform_pose(
            translation, rotation, marker['position'], marker['orientation'])

        forward = position[self.view_axis]
        if forward <= 0.0:                      # 카메라 뒤쪽
            return None

        distance = math.sqrt(sum(c * c for c in position))
        if not (self.min_range <= distance <= self.max_range):
            return None

        if abs(math.atan2(position[self.h_axis], forward)) > self.hfov / 2.0:
            return None
        if abs(math.atan2(position[self.v_axis], forward)) > self.vfov / 2.0:
            return None

        # 마커가 카메라를 향하고 있는가: 마커 법선과 '마커 -> 카메라' 방향의 각도.
        normal = quat_rotate_vector(quat_normalize(orientation), (0.0, 0.0, 1.0))
        to_camera = tuple(-c / distance for c in position)
        cos_angle = sum(n * v for n, v in zip(normal, to_camera))
        if cos_angle < math.cos(self.max_view_angle):
            return None

        return position, orientation

    # ------------------------------------------------------------------
    def _publish_visualization(self, results):
        """물체와 카메라 시야를 RViz 에 그린다. results 는 _tick 의 판정 결과."""
        stamp = self.get_clock().now().to_msg()

        for index, (marker, seen) in enumerate(results):
            visible = seen is not None
            # 보이면 초록, 안 보이면 회색 — Camera-Ready 자세 튜닝의 즉각적인 피드백
            color = (0.2, 0.85, 0.3) if visible else (0.45, 0.45, 0.45)
            normal = quat_rotate_vector(
                quat_normalize(marker['orientation']), (0.0, 0.0, 1.0))
            height = self.object_size[2]

            body = Marker()
            body.header.frame_id = self.base_frame
            body.header.stamp = stamp
            body.ns = 'sim_objects'
            body.id = index * 10
            body.type = Marker.CUBE
            body.action = Marker.ADD
            # 마커는 상자 윗면에 붙어 있으므로 몸통 중심은 법선 반대로 h/2 만큼
            body.pose = _make_pose(
                tuple(p - n * height / 2.0 for p, n in zip(marker['position'], normal)),
                marker['orientation'])
            body.scale.x, body.scale.y, body.scale.z = (float(v) for v in self.object_size)
            body.color.r, body.color.g, body.color.b, body.color.a = (*color, 0.85)
            self._viz_pub.publish(body)

            face = Marker()
            face.header = body.header
            face.ns = 'sim_objects'
            face.id = index * 10 + 1
            face.type = Marker.CUBE
            face.action = Marker.ADD
            face.pose = _make_pose(marker['position'], marker['orientation'])
            face.scale.x = face.scale.y = float(self.marker_size)
            face.scale.z = 0.002
            face.color.r = face.color.g = face.color.b = 0.95
            face.color.a = 1.0
            self._viz_pub.publish(face)

            label = Marker()
            label.header = body.header
            label.ns = 'sim_objects'
            label.id = index * 10 + 2
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose = _make_pose(
                tuple(p + n * 0.06 for p, n in zip(marker['position'], normal)),
                (0.0, 0.0, 0.0, 1.0))
            label.text = f"{marker['name']} id={marker['id']} " \
                         f"{'VISIBLE' if visible else 'HIDDEN'}"
            label.scale.z = 0.022
            label.color.r, label.color.g, label.color.b, label.color.a = (*color, 1.0)
            self._viz_pub.publish(label)

        self._publish_fov(stamp)

    def _publish_fov(self, stamp):
        """카메라 시야를 사각뿔 외곽선으로 그린다 (camera_frame 기준)."""
        reach = min(self.max_range, 0.5)
        th = math.tan(self.hfov / 2.0) * reach
        tv = math.tan(self.vfov / 2.0) * reach

        corners = []
        for sh, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            vec = [0.0, 0.0, 0.0]
            vec[self.view_axis] = reach
            vec[self.h_axis] = sh * th
            vec[self.v_axis] = sv * tv
            corners.append(Point(x=vec[0], y=vec[1], z=vec[2]))

        fov = Marker()
        fov.header.frame_id = self.camera_frame
        fov.header.stamp = stamp
        fov.ns = 'camera_fov'
        fov.id = 0
        fov.type = Marker.LINE_LIST
        fov.action = Marker.ADD
        fov.pose.orientation.w = 1.0
        fov.scale.x = 0.003
        fov.color.r, fov.color.g, fov.color.b, fov.color.a = 1.0, 0.75, 0.1, 0.7

        origin = Point(x=0.0, y=0.0, z=0.0)
        for i, corner in enumerate(corners):
            fov.points.append(origin)          # 꼭짓점에서 모서리로
            fov.points.append(corner)
            fov.points.append(corner)          # 밑면 사각형
            fov.points.append(corners[(i + 1) % 4])
        self._viz_pub.publish(fov)


def _make_pose(position, orientation):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in position)
    (pose.orientation.x, pose.orientation.y,
     pose.orientation.z, pose.orientation.w) = (float(v) for v in orientation)
    return pose


def main(args=None):
    rclpy.init(args=args)
    node = FakeArucoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
