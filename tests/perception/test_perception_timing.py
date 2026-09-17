import pytest
from perception_timing import FrameTiming, TimingWindow


def test_input_processing_and_publication_rates_are_distinct():
    window = TimingWindow()
    a, b, c = FrameTiming(), FrameTiming(), FrameTiming()
    a.outcome = c.outcome = 'published'
    a.ms = {'total': 10., 'capture_to_publish': 100.}
    b.ms = {'total': 20.}
    c.ms = {'total': 10., 'capture_to_publish': 120.}
    window.frames.extend([(0., a), (.1, b), (.2, c)])
    window.inputs.extend([-.1, 0., .05, .1, .15, .2, .25])
    report = window.summary()
    assert report['rgbd_input_hz'] == 20.
    assert report['processed_hz'] == 10.
    assert report['published_pick_hz'] == 5.
    assert report['stage_ms']['capture_to_publish']['mean'] == 110.
    assert report['stage_ms']['capture_to_publish']['samples'] == 2


def test_failed_stage_still_records_time():
    frame = FrameTiming()
    with pytest.raises(ValueError):
        with frame.stage('pose'):
            raise ValueError('bad input')
    frame.finish()
    assert 0 <= frame.ms['pose'] <= frame.ms['total']
