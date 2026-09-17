"""Bounded wall-clock profiling; no CUDA synchronization or disk I/O."""
from collections import Counter, deque
from contextlib import contextmanager
import time
import numpy as np


class FrameTiming:
    def __init__(self):
        self.started = time.perf_counter()
        self.ms = {}
        self.counts = Counter()
        self.outcome = 'rejected'

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = self.ms.get(name, 0.) + (time.perf_counter()-start)*1000.

    def finish(self):
        self.ms['total'] = (time.perf_counter()-self.started)*1000.
        return self


class TimingWindow:
    def __init__(self, size=120):
        self.frames = deque(maxlen=size)
        self.inputs = deque(maxlen=size*8)
        self.replaced_inputs = 0

    def input(self, replaced=False):
        self.inputs.append(time.monotonic())
        self.replaced_inputs += int(replaced)

    def add(self, frame):
        self.frames.append((time.monotonic(), frame))

    def summary(self):
        frames = list(self.frames)
        if not frames:
            return {'frames': 0}
        elapsed = frames[-1][0]-frames[0][0]
        keys = sorted({k for _, f in frames for k in f.ms})
        stages = {}
        for key in keys:
            # Latency and GPU events exist only for frames that reached them.
            optional = key == 'capture_to_publish' or key.endswith('_cuda_event')
            values = [f.ms[key] for _, f in frames if key in f.ms] if optional else [f.ms.get(key, 0.) for _, f in frames]
            stages[key] = dict(mean=round(float(np.mean(values)), 3),
                               p50=round(float(np.median(values)), 3),
                               p95=round(float(np.percentile(values, 95)), 3),
                               samples=len(values))
        inputs = [t for t in self.inputs if frames[0][0] <= t <= frames[-1][0]]
        counts = Counter()
        for _, f in frames: counts.update(f.counts)
        published = sum(f.outcome == 'published' for _, f in frames[1:])
        return {'frames': len(frames), 'window_s': round(elapsed, 3),
                'processed_hz': round((len(frames)-1)/elapsed, 2) if elapsed else 0.,
                'published_pick_hz': round(published/elapsed, 2) if elapsed else 0.,
                'rgbd_input_hz': round((len(inputs)-1)/(inputs[-1]-inputs[0]), 2) if len(inputs)>1 and inputs[-1]>inputs[0] else 0.,
                'replaced_inputs_total': self.replaced_inputs,
                'outcomes': dict(Counter(f.outcome for _, f in frames)),
                'counts': dict(counts), 'stage_ms': stages}
