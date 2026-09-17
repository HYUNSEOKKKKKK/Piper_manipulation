"""Keep a cuboid's long edge aligned, accounting for how the tool holds it."""
import math

from piper_pnp.geometry import (
    orientation_candidates, quat_from_rpy, quat_inverse, quat_multiply,
    quat_normalize, quat_rotate_vector, yaw_of,
)


def held_box_orientation(tool_q, pick_tool_q, pick_box_q):
    """Predict the held box frame using a rigid box-to-tool rotation."""
    return quat_normalize(quat_multiply(
        tool_q, quat_multiply(quat_inverse(pick_tool_q), pick_box_q)))


def long_axis_error(box_q, target_q):
    """Horizontal long-edge angle, modulo 180 degrees; vertical is invalid."""
    a = quat_rotate_vector(box_q, (1., 0., 0.))
    b = quat_rotate_vector(target_q, (1., 0., 0.))
    norm = math.hypot(*a[:2]) * math.hypot(*b[:2])
    if norm < 1e-9:
        return math.inf
    return math.acos(min(1., abs(a[0]*b[0] + a[1]*b[1]) / norm))


def aligned_place_candidates(pick_box_q, pick_tool_q, target_q, tilts, azimuths):
    """Try level boxes first, then tilted boxes with the same long-edge heading.

    Symmetry is applied to the BOX frame, before its fixed tool rotation.
    Rotating the TOOL locally by 90 degrees would exchange long/short edges.
    A tilted candidate's projected long edge is corrected to the target heading.
    """
    relative = quat_multiply(quat_inverse(pick_box_q), pick_tool_q)
    out, seen = [], set()
    for box_q in orientation_candidates(target_q, sorted(set(tilts)), azimuths, (0., math.pi)):
        correction = (yaw_of(target_q)-yaw_of(box_q)+math.pi/2) % math.pi-math.pi/2
        box_q = quat_multiply(quat_from_rpy(0., 0., correction), box_q)
        tool_q = quat_normalize(quat_multiply(box_q, relative))
        key = tuple(round(v, 8) for v in tool_q)
        neg = tuple(round(-v, 8) for v in tool_q)
        if key not in seen and neg not in seen:
            seen.add(key)
            out.append(tool_q)
    return out
