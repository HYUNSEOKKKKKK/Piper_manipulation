"""자동 연속 이송 또는 박스 투입마다 승인하는 반복 이송."""
from geometry_msgs.msg import Pose
from rcl_interfaces.srv import GetParameters


class BatchControlMixin:
    def _check_batch_bridge(self):
        mode = 'feed' if self._batch_workspace.operator_feed else 'batch'
        if self._batch_workspace.feed_drop:
            mode = 'feed-auto'
        if self._batch_workspace.sweep:
            mode = 'sweep'
        client = self.create_client(GetParameters, '/cuboid_pose_bridge/get_parameters',
                                    callback_group=self._cb_group)
        try:
            if not client.wait_for_service(timeout_sec=5.):
                self.get_logger().error(f'반복 인식 브리지가 없습니다. 터미널 2에서 run_bridge.sh {mode}를 실행하세요.')
                return False
            catalog=getattr(self,'_object_catalog',None)
            names=['batch_workspace_digest', 'target_color', 'dims']
            if catalog:names.append('object_catalog_digest')
            future = client.call_async(GetParameters.Request(names=names))
            response = self._await_control_future(future, 5.)
            values = response.values if response is not None else []
            valid = (len(values) == len(names) and values[0].type == 4 and values[1].type == 4
                     and values[2].type == 8
                     and values[0].string_value == self._batch_workspace.digest
                     and values[1].string_value == 'any'
                     and tuple(values[2].double_array_value) == self._batch_workspace.dims)
            if valid and catalog:
                valid=values[3].type==4 and values[3].string_value==catalog.digest
            if not valid:
                self.get_logger().error(f'반복 모드의 작업 영역/치수/색 설정이 브리지와 다릅니다. run_bridge.sh {mode}를 재실행하세요.')
            return valid
        finally:
            self.destroy_client(client)

    def _run_batch(self):
        if not self._check_batch_bridge():
            return False
        # MoveIt도 재시작됐을 수 있으므로 이미 놓은 박스의 충돌 모델을 복원한다.
        if self._batch_workspace.feed_drop:
            # Bound the existing goal before initial home/camera movement.
            if not self._sweep_obstacle(False, self._batch_workspace.table_z + self._batch_workspace.drop_max_height):
                return False
        else:
            for index in range(self._batch_completed):
                if not self._add_placed_box(index):
                    return False
        continuous = getattr(self, 'continuous_feed', False)
        feed = self._batch_workspace.operator_feed and not continuous
        approval_message = (
            '박스를 집기 영역에 놓고 손을 뺀 후 c를 누르면 9단계 한 사이클이 자동 진행됩니다. '
            '완료 후 홈에서 다음 박스 투입 승인을 기다립니다.' if feed else
            'c 한 번으로 이후 단계와 다음 박스 이송이 자동 진행됩니다.')
        if continuous:
            approval_message += (
                ' 첫 시작에만 홈을 거치고, 이후 촬영 자세에서 다음 박스를 새로 인식합니다. '
                '박스가 안 보이면 촬영 자세로 계속 대기하며, 다시 보이면 자동 이송합니다.')
        capacity_message = (
            f'중앙 한 곳 {self._batch_workspace.place(0)}에서 방향 제약·하강 없이 개방, '
            f'최대 {len(self._batch_workspace.places)}회.' if self._batch_workspace.feed_drop else
            f'놓기 자리 {len(self._batch_workspace.places)}개.')
        self.get_logger().info(
            f'연속 이송 준비: {capacity_message} '
            f'{approval_message} Space는 정지.')
        if not self._wait_for_confirmation(
                '박스 투입 완료·손 뺌 확인 / 한 사이클 시작' if feed else '연속 이송 전체 시작',
                timeout=None if (feed or continuous) else self.step_confirm_timeout):
            return False
        from_camera = False
        while not self.aborted and not self._terminate.is_set():
            if self._batch_completed >= len(self._batch_workspace.places):
                self.get_logger().info(
                    '설정한 최대 이송 횟수를 완료했습니다. 촬영 자세에서 대기합니다.'
                    if self._batch_workspace.feed_drop else
                    '놓기 자리가 모두 사용됐습니다. 현재 자세에서 대기합니다. 자리를 재사용하지 않습니다.')
                while not self.aborted and not self._terminate.wait(.2):
                    pass
                return True
            self._reset_cycle_state()
            destination = self._batch_workspace.place(self._batch_completed)
            self.get_logger().info(
                f'연속 이송 {self._batch_completed+1}: 새 박스 탐색, 놓기 자리 {destination}')
            if continuous:
                ok = self._run_cycle(from_camera=from_camera, return_home=False)
            else:
                ok = self._run_cycle()
            self.print_report()
            if not ok or self.aborted:
                self.emergency_stop('연속 이송 중단 — h → y로 새 실험을 준비하세요.', terminate=False)
                return False
            index = self._batch_completed
            # 놓은 뒤 후퇴와 촬영 자세 복귀가 끝난 물체를 다음 경로의 장애물로 반영한다.
            if not self._add_placed_box(index):
                self.emergency_stop('놓인 박스의 충돌 모델 추가 실패', terminate=False)
                return False
            self._batch_progress.finish()
            self._batch_completed = self._batch_progress.completed
            self.get_logger().info(f'이송 동작 {self._batch_completed}회 완료.')
            from_camera = continuous
            if feed and self._batch_completed < len(self._batch_workspace.places):
                self.get_logger().info(
                    '홈에서 박스 추가를 기다립니다. goal의 박스는 그대로 두세요. '
                    '새 박스를 집기 영역에 놓고 손을 뺀 후 c를 누르세요.')
                if not self._wait_for_confirmation('다음 박스 투입 완료·손 뺌 확인', timeout=None):
                    return False
        return False

    def _add_placed_box(self, index):
        if self._batch_workspace.feed_drop:
            # The shared pile bound was already installed before release.
            return True
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(
            float, self._batch_workspace.obstacle_center(index))
        pose.orientation.w = 1.
        return self.moveit.add_collision_box(
            f'batch_placed_{self._batch_workspace.digest[:12]}_{index}', self.base_frame,
            pose, self._batch_workspace.obstacle_size)

    def _set_batch_place_locked(self):
        workspace = getattr(self, '_batch_workspace', None)
        if workspace is not None and self._batch_completed < len(workspace.places):
            self._targets['place'] = {'position': workspace.place(self._batch_completed),
                                      'orientation': (0., 0., 0., 1.)}
            self._target_stamps['place'] = self.get_clock().now().nanoseconds
