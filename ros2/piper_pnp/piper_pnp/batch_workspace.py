"""연속 이송의 작업 영역과 서로 다른 놓기 자리. ROS/하드웨어와 무관한 설정 검증."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time


class BatchWorkspace:
    def __init__(self, config):
        self.sweep = config.get('motion_mode', 'precise') == 'sweep_drop'
        self.feed_drop = config.get('motion_mode', 'precise') == 'feed_drop'
        self.drop = self.sweep or self.feed_drop
        if config.get('motion_mode', 'precise') not in ('precise', 'sweep_drop', 'feed_drop'):
            raise ValueError('Unknown motion_mode')
        self.operator_feed = config.get('operator_feed', False)
        if type(self.operator_feed) is not bool:
            raise ValueError('operator_feed는 true 또는 false여야 합니다.')
        self.pick_bounds = tuple(config['pick_bounds_xy'])
        self.places = tuple(tuple(p) for p in config['place_positions_m'])
        self.exclusion_radius = float(config['place_exclusion_radius_m'])
        self.obstacle_size = tuple(config['placed_box_envelope_m'])
        self.table_z = float(config['table_z_m'])
        self.dims = tuple(config['object_dims_m'])
        self.previous_state_file = config.get('previous_state_file', '')
        self.previous_digests = tuple(config.get('previous_workspace_digests', []))
        if any(not isinstance(d, str) or len(d) != 64 or
               any(c not in '0123456789abcdef' for c in d) for d in self.previous_digests):
            raise ValueError('Invalid previous workspace digest')
        if (not isinstance(self.previous_state_file, str) or
                (self.previous_state_file and Path(self.previous_state_file).name != self.previous_state_file)):
            raise ValueError('previous_state_file must be a filename in the workspace directory')
        if self.drop:
            self.drop_size = tuple(config['drop_zone_size_xy_m'])
            self.drop_max_height = float(config['drop_max_height_m'])
            self.drop_clearance = float(config['drop_clearance_m'])
            count = config['max_transfers']
            if (self.operator_feed != self.feed_drop or len(self.places) != 1 or type(count) is not int
                    or not 1 <= count <= 10 or len(self.drop_size) != 2
                    or not all(type(v) in (int, float) and math.isfinite(v) and .15 <= v <= .5
                               for v in self.drop_size)
                    or not .05 <= self.drop_max_height <= .30
                    or not .04 <= self.drop_clearance <= .10):
                raise ValueError('Invalid sweep drop zone/capacity/height')
            self.places = self.places * count
        vectors = [(self.pick_bounds, 4), (self.obstacle_size, 3), (self.dims, 3)]
        vectors += [(p, 3) for p in self.places]
        if not self.places or not all(len(v) == n and all(
                isinstance(x, (int, float)) and math.isfinite(x) for x in v)
                for v, n in vectors):
            raise ValueError('작업 영역/놓기 위치/치수는 유한한 좌표여야 합니다.')
        x0, x1, y0, y1 = self.pick_bounds
        if x0 >= x1 or y0 >= y1:
            raise ValueError('집기 영역의 최소값은 최대값보다 작아야 합니다.')
        if (not math.isfinite(self.table_z) or not math.isfinite(self.exclusion_radius)
                or self.exclusion_radius <= 0
                or min(self.dims) <= 0 or min(self.obstacle_size) <= 0):
            raise ValueError('높이, 제외 반경과 치수 설정이 잘못됐습니다.')
        if self.exclusion_radius < math.hypot(*self.obstacle_size[:2]) / 2:
            raise ValueError('놓기 제외 반경이 놓인 물체의 보호 영역보다 작습니다.')
        for i, p in enumerate(self.places):
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                raise ValueError('놓기 자리가 집기 영역 안에 있습니다.')
            for q in self.places[:i]:
                if (not self.drop and abs(p[0]-q[0]) < self.obstacle_size[0] + .01 and
                        abs(p[1]-q[1]) < self.obstacle_size[1] + .01):
                    raise ValueError('놓기 자리의 보호 영역이 겹칩니다. 간격을 늘리세요.')
        if self.drop:
            x, y, z = self.places[0]
            dx, dy = self.drop_size
            if not (x+dx/2 <= x0 or x-dx/2 >= x1 or y+dy/2 <= y0 or y-dy/2 >= y1):
                raise ValueError('Drop zone overlaps pick region')
            if self.feed_drop and not self.table_z + .10 <= z <= self.table_z + .40:
                raise ValueError('Fixed drop TCP height must be 0.10–0.40 m above the table')
        payload = json.dumps(config, sort_keys=True, separators=(',', ':'))
        self.digest = hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def allows_pick(self, point):
        if len(point) != 3 or not all(math.isfinite(x) for x in point):
            return False
        x, y, _ = point
        x0, x1, y0, y1 = self.pick_bounds
        return (x0 <= x <= x1 and y0 <= y <= y1 and all(
            math.hypot(x-p[0], y-p[1]) > self.exclusion_radius for p in self.places))

    def place(self, index):
        # precise는 서로 다른 자리, drop은 같은 좌표와 별도의 이송 횟수 한도다.
        if not 0 <= index < len(self.places):
            raise IndexError('빈 놓기 자리가 없습니다.')
        return self.places[index]

    def obstacle_center(self, index):
        x, y, _ = self.place(index)
        return x, y, self.table_z + self.obstacle_size[2] / 2


class BatchProgress:
    """재시작해도 이미 쓴 놓기 자리를 재사용하지 않는 작은 진행 기록."""
    def __init__(self, config_path, workspace, *, allow_interrupted=False):
        self.path = Path(str(Path(config_path).resolve()) + '.state.json')
        self.workspace = workspace
        self.completed = 0
        self.in_progress = False
        if not self.path.exists() and workspace.previous_state_file:
            previous = self.path.parent / workspace.previous_state_file
            if previous.exists():
                state = json.loads(previous.read_text())
                if (type(state.get('completed')) is not int or state['completed'] < 0
                        or type(state.get('in_progress')) is not bool):
                    raise ValueError(f'이전 실험 기록을 확인할 수 없습니다: {previous}')
                # Changing placement policy must not bypass a held box or occupied
                # goal. Keep the old record untouched; normal h→y handles reset.
                self.in_progress = state['in_progress'] or state['completed'] > 0
                if self.in_progress and not allow_interrupted:
                    raise ValueError('이전 feed 기록이 남아 있습니다. 빈 그리퍼·goal 확인 후 h → y.')
        if self.path.exists():
            state = json.loads(self.path.read_text())
            previous_config = (allow_interrupted and workspace.feed_drop and
                               state.get('workspace_digest') in workspace.previous_digests)
            if ((state.get('workspace_digest') != workspace.digest and not previous_config)
                    or type(state.get('completed')) is not int
                    or not 0 <= state['completed'] <= len(workspace.places)
                    or type(state.get('in_progress')) is not bool
                    or (state['in_progress'] and not allow_interrupted)):
                raise ValueError(
                    f'이전 이송이 중단됐거나 작업 영역이 바뀌었습니다. 팔과 놓기 자리를 확인한 후 '
                    f'진행 기록을 정리해야 합니다: {self.path}')
            self.completed = state['completed']
            self.in_progress = state['in_progress']
            if previous_config:
                # Keep the original bytes/count until operator-confirmed reset.
                # In particular, changing only height must not bypass an old stop.
                self._reset_source_state = state
                self.in_progress = True

    def reset_after_home(self):
        """Operator cleared all boxes and home was verified; back up before reset."""
        if self.path.exists():
            data = self.path.read_bytes()
            expected = getattr(self, '_reset_source_state', dict(
                workspace_digest=self.workspace.digest,
                completed=self.completed, in_progress=self.in_progress))
            if json.loads(data) != expected:
                raise ValueError('진행 기록이 외부에서 바뀌었습니다. 초기화를 중단합니다.')
            backup = self.path.with_name(self.path.name + f'.before-reset-{time.time_ns()}')
            with backup.open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        previous = self.completed, self.in_progress
        self.completed, self.in_progress = 0, False
        try:
            self._save()
        except Exception:
            self.completed, self.in_progress = previous
            raise
        if hasattr(self, '_reset_source_state'):
            del self._reset_source_state

    def _save(self):
        state = dict(workspace_digest=self.workspace.digest,
                     completed=self.completed, in_progress=self.in_progress)
        fd, temp_path = tempfile.mkstemp(prefix='.batch-progress-', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(state, stream, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, self.path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def reserve(self):
        if self.in_progress or self.completed >= len(self.workspace.places):
            raise ValueError('새 이송을 시작할 빈 자리가 없습니다.')
        self.in_progress = True
        self._save()

    def finish(self):
        if not self.in_progress:
            raise ValueError('시작하지 않은 이송을 완료할 수 없습니다.')
        self.completed += 1
        self.in_progress = False
        self._save()
