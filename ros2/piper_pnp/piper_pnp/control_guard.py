"""ROS와 무관한 실기 명령 게이트. 시작 시 닫히고, 시드 정합성 확인 후 열린다."""
import math
import threading


class CommandGuard:
    def __init__(self, joints, max_age=0.25, tolerance=0.03):
        self.joints = tuple(joints)
        self.max_age = max_age
        self.tolerance = tolerance
        self.lock = threading.RLock()
        self.enabled = False
        self.feedback = self.command = None
        self.feedback_at = self.command_at = None

    def record(self, kind, names, values, now):
        with self.lock:
            if len(names) != len(values) or len(set(names)) != len(names):
                return False
            row = dict(zip(names, values))
            if not all(j in row and math.isfinite(row[j]) for j in self.joints):
                return False
            if not all(math.isfinite(v) for v in row.values()):
                return False
            setattr(self, kind, row)
            setattr(self, kind+'_at', now)
            return True

    def reason(self, now, require_alignment=False):
        with self.lock:
            for kind in ('feedback', 'command'):
                received = getattr(self, kind+'_at')
                if received is None or not 0 <= now-received <= self.max_age:
                    return f'{kind} is missing or stale'
            if require_alignment:
                for name in self.joints:
                    tol = 0.003 if name == 'gripper' else self.tolerance
                    if abs(self.command[name]-self.feedback[name]) > tol:
                        return f'{name}: command is not synchronized to feedback'
            return None

    def open(self, now):
        with self.lock:
            reason = self.reason(now, require_alignment=True)
            if reason:
                self.enabled = False
                raise ValueError(reason)
            self.enabled = True

    def close(self):
        with self.lock:
            self.enabled = False
