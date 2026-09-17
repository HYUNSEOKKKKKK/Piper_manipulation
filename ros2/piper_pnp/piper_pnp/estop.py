"""
piper_pnp_estop — 키보드 비상 정지 콘솔.

전용 터미널에서 띄워두고 그 창에 포커스를 둔 채 작업한다. 아무 키나 누르면
즉시 정지한다.

    ros2 run piper_pnp piper_pnp_estop

키 배치
-------
    c        다음 단계 진행 (step_confirm 모드에서 승인)
    h → y    빈 그리퍼·goal 전체 비움 확인 후 홈 복귀 / 이송 횟수 0
    r        기존 단일 단계 정지 해제 요청
    q        이 콘솔 종료 (팔은 건드리지 않는다)
    그 외    즉시 정지

당황해서 키보드를 아무렇게나 두드려도 정지하도록, 지정된 키를 제외한
모든 입력이 정지로 간다.

정지 체인
---------
    1) control_enable(false)   게이트를 먼저 닫아 새 명령이 팔에 가지 않게 한다
    2) emergency_stop          현재 자세를 붙잡는다 (전원 유지 — 팔이 안 떨어진다)
    3) 컨트롤러 ~/stop         실행 중인 궤적 goal 취소 (best effort)

1)과 2)를 이 노드가 **직접** 호출하는 것이 중요하다. 컨트롤러가 멈춰 있거나
응답하지 않아도 정지가 동작해야 하기 때문이다. 3)은 되면 좋고 안 돼도 무방하다.

[주의] enable_agx_arm(false) 는 절대 쓰지 않는다. 토크가 풀려 팔이 자중으로
떨어진다. 브레이크 없는 팔의 안전 상태는 '전원 꺼짐'이 아니라 '전원이 켜진 채
버티는 상태'다.

[한계] 이 터미널에 포커스가 있어야 동작한다. 소프트웨어 전체가 멎는 상황에서는
아무것도 할 수 없다. 물리 버튼의 대체가 아니라 그 전 단계다.

[주의] ros2 launch 로 띄우면 키보드를 읽을 수 없다 (stdin 이 터미널이 아니다).
반드시 ros2 run 으로 별도 터미널에서 실행할 것.
"""

import sys
import termios
import threading
import tty

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Empty, SetBool, Trigger
from piper_pnp.console_input import TerminalKeyReader, describe_key, ConsoleKeyState

BANNER = """
==============================================================
  PiPER 비상 정지 콘솔 — 이 창에 포커스를 두세요
--------------------------------------------------------------
   c / ㅊ   단계 승인 / feed-auto·batch·sweep: 연속 / feed: 한 사이클
   h / ㅗ   정지 후 처음부터 준비 (안내 확인 후 y / ㅛ)
   r / ㄱ   기존 단일 단계 정지 해제 요청
   q / ㅂ   이 콘솔만 종료
   그 외    >>> 즉시 정지 <<<
--------------------------------------------------------------
  c는 Enter 없이 누릅니다. Space와 Enter는 모두 정지 키입니다.
  한/영 상태와 무관하게 같은 물리 키가 동작합니다.
==============================================================
"""

# 2벌식 키보드에서 한/영 전환을 놓쳤을 때 나오는 호환 자모.
#   c -> ㅊ (U+314A) / r -> ㄱ (U+3131) / q -> ㅂ (U+3142)


class EstopConsole(Node):
    def __init__(self):
        super().__init__('piper_pnp_estop')
        self._cb_group = ReentrantCallbackGroup()

        self.declare_parameter('control_enable_service', '/control_enable')
        self.declare_parameter('emergency_stop_service', '/emergency_stop')
        self.declare_parameter('controller_namespace', '/piper_pnp_controller')

        self.control_enable_name = self.get_parameter('control_enable_service').value
        self.emergency_stop_name = self.get_parameter('emergency_stop_service').value
        controller = self.get_parameter('controller_namespace').value.rstrip('/')

        self._control_enable = self.create_client(
            SetBool, self.control_enable_name, callback_group=self._cb_group)
        self._emergency_stop = self.create_client(
            Empty, self.emergency_stop_name, callback_group=self._cb_group)
        self._stop = self.create_client(
            Trigger, f'{controller}/stop', callback_group=self._cb_group)
        self._resume = self.create_client(
            Trigger, f'{controller}/resume', callback_group=self._cb_group)
        self._continue = self.create_client(
            Trigger, f'{controller}/continue', callback_group=self._cb_group)
        self._reset = self.create_client(
            SetBool, f'{controller}/reset_experiment', callback_group=self._cb_group)
        self.create_subscription(
            String, f'{controller}/experiment_state',
            lambda msg: self.get_logger().info(msg.data),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE), callback_group=self._cb_group)

        self._stopped = False

    # ------------------------------------------------------------------
    def trigger_stop(self, reason='정지 요청'):
        """정지 체인. 응답을 기다리지 않는다 — 비상 상황에서 블로킹하면 안 된다."""
        self.get_logger().warn('*' * 58)
        self.get_logger().warn('  정 지')
        self.get_logger().warn(f'  원인: {reason}')
        self.get_logger().warn('*' * 58)

        if self._control_enable.service_is_ready():
            request = SetBool.Request()
            request.data = False
            self._control_enable.call_async(request)
            self.get_logger().warn(f'  1) {self.control_enable_name}(false) — 게이트 차단')
        else:
            self.get_logger().warn(
                f'  1) {self.control_enable_name} 없음 (시뮬레이션이면 정상)')

        if self._emergency_stop.service_is_ready():
            self._emergency_stop.call_async(Empty.Request())
            self.get_logger().warn(f'  2) {self.emergency_stop_name} — 현재 자세 유지')
        else:
            self.get_logger().warn(
                f'  2) {self.emergency_stop_name} 없음 (시뮬레이션이면 정상)')

        if self._stop.service_is_ready():
            self._stop.call_async(Trigger.Request())
            self.get_logger().warn('  3) 컨트롤러에 궤적 취소 요청')
        else:
            self.get_logger().warn('  3) 컨트롤러 정지 서비스 없음')

        self._stopped = True
        self.get_logger().warn('  처음부터 다시 준비하려면 h → 안내 확인 후 y. Enter는 누르지 마세요.')

    def trigger_reset_prompt(self):
        self.trigger_stop('홈 초기화 준비')
        self.get_logger().warn(
            '초기화 확인: 그리퍼가 비었고, goal의 박스를 전부 치웠으며, '
            '팔의 홈 복귀 경로에서 손과 장애물을 뺐는지 확인하세요.')
        self.get_logger().warn(
            '확인했으면 y / ㅛ: 홈 복귀 후 이송 횟수 0. 그 외 키: 취소/정지. '
            '박스를 쥐고 있다면 y를 누르지 마세요. 그리퍼를 자동으로 열지 않습니다.')

    def trigger_reset(self):
        if not self._reset.service_is_ready():
            self.get_logger().warn(
                '초기화 서비스가 없습니다. 수정된 터미널 1을 한 번 다시 실행하세요.')
            return
        request = SetBool.Request(data=True)
        future = self._reset.call_async(request)
        future.add_done_callback(self._show_control_response)

    def trigger_resume(self):
        if self._resume.service_is_ready():
            # cuboid 컨트롤러는 다음 사이클의 관절 동기화 후에만 게이트를 연다.
            future = self._resume.call_async(Trigger.Request())
            future.add_done_callback(self._show_control_response)
        else:
            self.get_logger().warn('컨트롤러가 없습니다. 제어 게이트는 닫힌 상태로 유지합니다.')

    def trigger_continue(self):
        if self._continue.service_is_ready():
            future = self._continue.call_async(Trigger.Request())
            future.add_done_callback(self._show_control_response)
        else:
            self.get_logger().warn(
                '컨트롤러 continue 서비스가 없습니다 '
                '(step_confirm 모드가 아니거나 컨트롤러가 안 떠 있음).')

    def _show_control_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(response.message)
            else:
                self.get_logger().warn(response.message)
        except Exception as exc:
            self.get_logger().warn(f'컨트롤러 응답 확인 실패: {exc}')


def main(args=None):
    if not sys.stdin.isatty():
        print('[ERROR] 표준 입력이 터미널이 아닙니다.\n'
              '        ros2 launch 로는 키보드를 읽을 수 없습니다.\n'
              '        별도 터미널에서 ros2 run piper_pnp piper_pnp_estop 으로 실행하세요.',
              file=sys.stderr)
        return 1

    rclpy.init(args=args)
    node = EstopConsole()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print(BANNER, flush=True)

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        # cbreak: Enter 없이 키 하나씩 즉시 읽는다.
        tty.setcbreak(fd)
        reader = TerminalKeyReader(fd)
        keys = ConsoleKeyState()
        while rclpy.ok():
            key = reader.read_key()
            if key is None:
                continue
            action = keys.action(key)
            if action == 'quit':
                print('\n콘솔을 종료합니다. (팔 상태는 그대로입니다)', flush=True)
                break
            if action == 'reset_prompt':
                node.trigger_reset_prompt()
            elif action == 'reset_confirm':
                node.trigger_reset()
            elif action == 'resume':
                node.trigger_resume()
            elif action == 'continue':
                node.trigger_continue()
            else:
                node.trigger_stop(f'키 입력 {describe_key(key)}')
    except KeyboardInterrupt:
        node.trigger_stop('Ctrl+C / KeyboardInterrupt')
    except (EOFError, OSError) as exc:
        node.trigger_stop(f'키보드 입력 연결 종료: {exc}')
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
