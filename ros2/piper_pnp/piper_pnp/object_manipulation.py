"""Rigid held-box geometry for size-aware placement. No ROS or hardware calls."""
import math
from piper_pnp.geometry import quat_from_rpy, quat_inverse, quat_multiply, quat_rotate_vector, transform_pose, yaw_of


def box_pose(pick):
    return transform_pose(pick['position'],pick['orientation'],
                          pick['box_in_grasp_position'],pick['box_in_grasp_orientation'])


def span(dims,q,direction):
    axes=[quat_rotate_vector(q,tuple(1. if i==j else 0. for i in range(3))) for j in range(3)]
    return sum(d*abs(sum(a*b for a,b in zip(axis,direction))) for d,axis in zip(dims,axes))


def grip_width(pick,tool_q):
    _,q=box_pose(pick)
    return span(pick['dims'],q,quat_rotate_vector(tool_q,(0.,1.,0.)))


def held_transform(pick,pick_tool_q,grasp_offset,contact=None):
    center,q=box_pose(pick)
    if contact is None:
        axis=quat_rotate_vector(pick_tool_q,(0.,0.,1.))
        contact=tuple(p-grasp_offset*a for p,a in zip(pick['position'],axis))
    inv=quat_inverse(pick_tool_q)
    return (quat_rotate_vector(inv,tuple(c-p for c,p in zip(center,contact))),
            quat_multiply(inv,q))


def level_place_candidates(pick,pick_tool_q,target_q):
    _,box_q=box_pose(pick)
    relative=quat_multiply(quat_inverse(box_q),pick_tool_q)
    # Catalog convention for new models: Z is the intended support thickness.
    yaw=yaw_of(target_q)
    angles=(0.,math.pi/2,math.pi,3*math.pi/2) if abs(pick['dims'][0]-pick['dims'][1])<1e-9 else (0.,math.pi)
    # Either sign of the unlabeled model Z axis describes the same support
    # faces. Include both, otherwise a valid 180-degree pose representative
    # could demand an upside-down tool at the destination.
    return [quat_multiply(quat_multiply(quat_from_rpy(0.,0.,yaw+a),
                                        quat_from_rpy(flip,0.,0.)),relative)
            for flip in (0.,math.pi) for a in angles]


def place_positions(pick,pick_tool_q,tool_q,place_xy,table_z,grasp_offset,retreat,contact=None):
    offset,relative=held_transform(pick,pick_tool_q,grasp_offset,contact)
    box_q=quat_multiply(tool_q,relative)
    height=span(pick['dims'],box_q,(0.,0.,1.))/2
    center=(place_xy[0],place_xy[1],table_z+height+pick['release_clearance_m'])
    offset=quat_rotate_vector(tool_q,offset)
    release=tuple(c-o for c,o in zip(center,offset))
    axis=quat_rotate_vector(tool_q,(0.,0.,1.))
    ready=tuple(p-retreat*a for p,a in zip(release,axis))
    return ready,release


def check_level_release(pick,pick_tool_q,tool_q,tool_position,target_q,table_z,grasp_offset,tolerance,contact=None):
    offset,relative=held_transform(pick,pick_tool_q,grasp_offset,contact)
    box_q=quat_multiply(tool_q,relative)
    normal=quat_rotate_vector(box_q,(0.,0.,1.))
    tilt=math.acos(min(1.,abs(normal[2])))
    period=math.pi/2 if abs(pick['dims'][0]-pick['dims'][1])<1e-9 else math.pi
    error=abs((yaw_of(box_q)-yaw_of(target_q)+period/2)%period-period/2)
    center_offset=quat_rotate_vector(tool_q,offset)
    bottom=tool_position[2]+center_offset[2]-span(pick['dims'],box_q,(0.,0.,1.))/2
    return tilt<=tolerance and error<=tolerance and table_z-.003<=bottom<=table_z+pick['release_clearance_m']+.015
