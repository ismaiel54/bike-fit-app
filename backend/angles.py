import math
from typing import Dict, Optional, Tuple


def compute_signed_segment_angle_deg(
    p_from: Tuple[float, float],
    p_to: Tuple[float, float],
) -> float:
    """Signed angle of the directed segment *p_from→p_to* from the +x axis.

    Input points are in **image** coordinates where y increases downward.
    We convert to standard maths coordinates (y increases upward) by negating dy.

    Returns:
        Angle in degrees in **[-180, 180]**, counter-clockwise positive.
    """
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    return math.degrees(math.atan2(-dy, dx))


def compute_torso_to_horizontal_deg(
    hip: Tuple[float, float],
    shoulder: Tuple[float, float],
    frame_rotation_deg: int = 0,
) -> Tuple[Optional[float], Dict[str, object]]:
    """Torso angle between torso segment and **true horizontal**.

    Definition:
        0° = torso perfectly horizontal (maximally aero).
        90° = torso perfectly vertical (standing upright).

    Notes:
        - Direction invariant: hip→shoulder vs shoulder→hip yields same result.
        - Rotation aware: if the frame was rotated clockwise before pose estimation,
          pass that rotation as ``frame_rotation_deg`` to un-rotate the torso vector
          back into the original camera orientation before measuring against horizontal.
    """
    dx_img = shoulder[0] - hip[0]
    dy_img = shoulder[1] - hip[1]

    length = math.sqrt(dx_img * dx_img + dy_img * dy_img)
    if length < 1e-6:
        return None, {"valid": False, "reason": "zero_length_vector"}

    signed_deg_rot = compute_signed_segment_angle_deg(hip, shoulder)

    # If the frame was rotated CW by rot, measured angle is (a - rot).
    # Undo back to the original camera orientation.
    signed_deg = signed_deg_rot + float(frame_rotation_deg)
    signed_deg = ((signed_deg + 180.0) % 360.0) - 180.0  # wrap to [-180, 180)

    abs_deg = abs(signed_deg)
    angle_to_horizontal = abs_deg if abs_deg <= 90.0 else (180.0 - abs_deg)

    # Product/UI convention requested: report as (90 - measured torso angle).
    # This maps large measured values (near vertical) to small reported values.
    reported_torso_angle = 90.0 - angle_to_horizontal
    reported_torso_angle = max(0.0, min(90.0, reported_torso_angle))

    # Reliability: shoulder above hip in the original image coords (y-down).
    # Rotate the vector back into the original image frame and check dy < 0.
    vmx, vmy = dx_img, -dy_img  # image→math coords
    theta = math.radians(frame_rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    vmx_o = cos_t * vmx - sin_t * vmy
    vmy_o = sin_t * vmx + cos_t * vmy
    dy_img_orig = -vmy_o
    shoulder_above_hip = dy_img_orig < 0

    diagnostics: Dict[str, object] = {
        "valid": True,
        "reliable": shoulder_above_hip,
        "shoulder_above_hip": shoulder_above_hip,
        "angle_to_horizontal_deg_raw": round(angle_to_horizontal, 3),
        "angle_reported_deg": round(reported_torso_angle, 3),
        "signed_deg_rot": round(signed_deg_rot, 3),
        "signed_deg_unrot": round(signed_deg, 3),
        "dx_img": round(dx_img, 4),
        "dy_img": round(dy_img, 4),
        "frame_rotation_deg": int(frame_rotation_deg),
    }
    return reported_torso_angle, diagnostics

