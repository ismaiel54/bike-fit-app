import math
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose

# Global state for temporal smoothing and rotation tracking
_smoothing_state = {
    "foot_angle_ema": None,
    "torso_angle_ema": None,
    "torso_debug": None,
    "display_rotation_deg": 0,
}

# ---------------------------------------------------------------------------
# Constants for the angle pipeline
# ---------------------------------------------------------------------------

MIN_LANDMARK_CONFIDENCE = 0.4

SIDE_LANDMARK_ENUMS: Dict[str, Dict[str, int]] = {
    "left": {
        "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "hip": mp_pose.PoseLandmark.LEFT_HIP,
        "knee": mp_pose.PoseLandmark.LEFT_KNEE,
        "ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
        "elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
        "wrist": mp_pose.PoseLandmark.LEFT_WRIST,
        "foot_index": mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    },
    "right": {
        "shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "hip": mp_pose.PoseLandmark.RIGHT_HIP,
        "knee": mp_pose.PoseLandmark.RIGHT_KNEE,
        "ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
        "elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
        "wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
        "foot_index": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
    },
}


# ---------------------------------------------------------------------------
# Side selection
# ---------------------------------------------------------------------------

def select_analysis_side(
    landmarks,
    min_confidence: float = MIN_LANDMARK_CONFIDENCE,
) -> Tuple[str, Dict[str, float]]:
    """Choose the body side with more reliable landmarks for side-on analysis.

    Evaluates visibility/confidence of key landmarks on each side, giving
    extra weight to the three joints critical for bike-fit angles (shoulder,
    hip, knee).

    Returns:
        (side, scores) where side is ``"left"`` or ``"right"`` and scores
        maps side names to their weighted-average confidence.
    """
    scores: Dict[str, float] = {}

    for side_name, enum_map in SIDE_LANDMARK_ENUMS.items():
        total = 0.0
        weight_sum = 0.0
        for joint_name, enum_val in enum_map.items():
            weight = 2.0 if joint_name in ("shoulder", "hip", "knee") else 1.0
            try:
                conf = landmarks.landmark[enum_val].visibility
            except (IndexError, AttributeError):
                conf = 0.0
            total += conf * weight
            weight_sum += weight
        scores[side_name] = total / weight_sum if weight_sum > 0 else 0.0

    chosen = "left" if scores.get("left", 0) >= scores.get("right", 0) else "right"
    return chosen, scores


# ---------------------------------------------------------------------------
# Landmark accessors with confidence
# ---------------------------------------------------------------------------

def _get_landmark_with_confidence(
    landmarks, landmark_enum,
) -> Tuple[float, float, float]:
    """Return ``(x, y, confidence)`` in normalised image coordinates."""
    lm = landmarks.landmark[landmark_enum]
    return (lm.x, lm.y, lm.visibility)


# ---------------------------------------------------------------------------
# Geometry helpers — explicit names, explicit coordinate handling
# ---------------------------------------------------------------------------

def compute_signed_segment_angle_deg(
    p_from: Tuple[float, float],
    p_to: Tuple[float, float],
) -> float:
    """Signed angle of the directed segment *p_from→p_to* from the +x axis.

    Converts from **image** coordinates (y increases downward) to standard
    maths coordinates (y increases upward) by negating dy.

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
    """Torso angle relative to **real-world horizontal** for side-on bike fit.

    Definition
    ----------
    0° = torso perfectly horizontal (maximally aero).
    90° = torso perfectly vertical (standing upright).

    Algorithm
    ---------
    1. Build the hip→shoulder vector in the (possibly rotated) image frame.
    2. If the frame was rotated by ``_normalize_frame_orientation`` (e.g.
       portrait 90° CW), un-rotate the vector so it is back in the
       original camera orientation where +x = real-world horizontal.
       This is a deterministic coordinate transform using a **known**
       rotation angle — not a heuristic.
    3. Compute ``angle = abs(atan2(dy, dx))`` in image coordinates (y-down)
       and take the acute angle to horizontal: ``min(angle, 180 - angle)``.
    4. Validate that the shoulder is above the hip; flag unreliable if not.

    Parameters
    ----------
    frame_rotation_deg :
        Degrees the frame was rotated **clockwise** before MediaPipe
        processed it (from ``_smoothing_state["display_rotation_deg"]``).
        Typical values: 0 (landscape, no rotation) or 90 (portrait → landscape).
    """
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]

    # Un-rotate the torso vector when the frame was rotated for normalisation.
    # The landmarks are in the rotated frame's coordinate space; this puts them
    # back in the original camera space where x = real-world horizontal.
    if frame_rotation_deg != 0:
        rad = math.radians(-frame_rotation_deg)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        dx, dy = cos_r * dx - sin_r * dy, sin_r * dx + cos_r * dy

    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return None, {"valid": False, "reason": "zero_length_vector"}

    # Angle to horizontal: abs(atan2(dy, dx)) gives the angle of the vector
    # from the +x axis in image coordinates (y-down).  Taking the acute angle
    # to the nearest horizontal axis (0° or 180°) gives the torso tilt.
    raw_angle_deg = abs(math.degrees(math.atan2(dy, dx)))
    angle_to_horizontal = min(raw_angle_deg, 180.0 - raw_angle_deg)
    angle_to_horizontal = max(0.0, min(90.0, angle_to_horizontal))

    # After un-rotation, shoulder above hip in the original image ↔ dy < 0.
    shoulder_above_hip = dy < 0
    reliable = shoulder_above_hip

    diagnostics: Dict[str, object] = {
        "valid": True,
        "reliable": reliable,
        "shoulder_above_hip": shoulder_above_hip,
        "raw_angle_deg": round(raw_angle_deg, 2),
        "dx": round(dx, 4),
        "dy": round(dy, 4),
        "frame_rotation_deg": frame_rotation_deg,
    }
    return angle_to_horizontal, diagnostics


def get_metric_reliability(
    landmarks,
    required_enums: List,
    min_confidence: float = MIN_LANDMARK_CONFIDENCE,
) -> Tuple[bool, float, List[str]]:
    """Check that every landmark in *required_enums* meets *min_confidence*.

    Returns:
        ``(is_reliable, mean_confidence, weak_landmark_names)``
    """
    confidences: List[float] = []
    weak: List[str] = []
    for enum_val in required_enums:
        try:
            conf = landmarks.landmark[enum_val].visibility
        except (IndexError, AttributeError):
            conf = 0.0
        confidences.append(conf)
        if conf < min_confidence:
            name = enum_val.name if hasattr(enum_val, "name") else str(enum_val)
            weak.append(name)

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return len(weak) == 0, mean_conf, weak


def _get_landmark_coords(
    landmarks: mp.solutions.pose.PoseLandmark, landmark_enum: mp.solutions.pose.PoseLandmark
) -> Tuple[float, float]:
    """Get normalized coordinates (0-1 range)."""
    landmark = landmarks.landmark[landmark_enum]
    return (landmark.x, landmark.y)


def _get_landmark_pixel_coords(
    landmarks: mp.solutions.pose.PoseLandmark,
    landmark_enum: mp.solutions.pose.PoseLandmark,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int]:
    """Get pixel coordinates from normalized coordinates."""
    landmark = landmarks.landmark[landmark_enum]
    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)
    return (x, y)


def _compute_angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Compute the angle ABC (with vertex at B) in degrees.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    ba_vec = np.array(ba)
    bc_vec = np.array(bc)

    dot_prod = np.dot(ba_vec, bc_vec)
    norm_prod = np.linalg.norm(ba_vec) * np.linalg.norm(bc_vec)
    if norm_prod == 0:
        return float("nan")

    cos_angle = np.clip(dot_prod / norm_prod, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _get_mid_torso_points(landmarks: mp.solutions.pose.PoseLandmark) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Get mid-hip and mid-shoulder points for stable torso measurement.
    
    Args:
        landmarks: MediaPipe pose landmarks
    
    Returns:
        (mid_shoulder_norm, mid_hip_norm) or (None, None) if insufficient visibility
    """
    try:
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Check visibility thresholds
        min_shoulder_vis = min(left_shoulder.visibility, right_shoulder.visibility)
        min_hip_vis = min(left_hip.visibility, right_hip.visibility)
        
        if min_shoulder_vis < 0.3 or min_hip_vis < 0.3:
            return None, None
        
        # Compute mid-points
        mid_shoulder = (
            (left_shoulder.x + right_shoulder.x) / 2.0,
            (left_shoulder.y + right_shoulder.y) / 2.0,
        )
        mid_hip = (
            (left_hip.x + right_hip.x) / 2.0,
            (left_hip.y + right_hip.y) / 2.0,
        )
        
        return mid_shoulder, mid_hip
    except (IndexError, AttributeError):
        return None, None


def _compute_torso_angle_deg(
    hip: Tuple[float, float], shoulder: Tuple[float, float]
) -> Tuple[float, float, float]:
    """
    Compute torso angle relative to horizontal.
    0° = perfectly horizontal (super low/aero), 90° = perfectly vertical.
    
    Uses: torso_angle_h = degrees(atan2(abs(dy), abs(dx)))
    
    Args:
        hip: (x, y) coordinates of mid-hip (normalized 0-1)
        shoulder: (x, y) coordinates of mid-shoulder (normalized 0-1)
    
    Returns:
        (angle_deg, dx, dy) where angle is in [0, 90] and dx/dy are vector components for debugging
    """
    # Define torso vector: mid_hip → mid_shoulder
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]  # Note: y increases downward in image coordinates
    
    # Check for zero vector
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return float("nan"), dx, dy
    
    # Compute angle to horizontal using atan2(abs(dy), abs(dx))
    # This directly returns angle-to-horizontal
    ang_rad = math.atan2(abs(dy), abs(dx))
    ang_deg = math.degrees(ang_rad)  # 0..90
    
    # Clamp to [0, 90] range
    torso_angle = max(0.0, min(90.0, ang_deg))
    
    return torso_angle, dx, dy


def _smooth_angle_with_ema(
    current_angle: Optional[float],
    previous_ema: Optional[float],
    alpha: float = 0.2,
    outlier_threshold: float = 15.0,
) -> Optional[float]:
    """
    Smooth angle using exponential moving average with outlier rejection.
    
    Args:
        current_angle: Current angle measurement (degrees)
        previous_ema: Previous EMA value
        alpha: Smoothing factor (0-1, lower = more smoothing)
        outlier_threshold: Maximum change allowed (degrees) - tightened to 15° for torso
    
    Returns:
        Smoothed angle or None if current is invalid
    """
    if current_angle is None or math.isnan(current_angle):
        return previous_ema  # Hold previous value if current is invalid
    
    if previous_ema is None:
        return current_angle  # First frame, no smoothing
    
    # Outlier rejection: if jump is too large, treat as outlier
    angle_diff = abs(current_angle - previous_ema)
    if angle_diff > outlier_threshold:
        # Log outlier for debugging
        print(f"Torso angle outlier rejected: {current_angle:.1f}° (previous: {previous_ema:.1f}°, diff: {angle_diff:.1f}°)")
        return previous_ema  # Keep previous smoothed value
    
    # Exponential moving average
    smoothed = alpha * current_angle + (1.0 - alpha) * previous_ema
    return smoothed


def _robust_foot_keypoint(
    landmarks,
    frame_width: int,
    frame_height: int,
    min_confidence: float = 0.5,
    foot_enum=None,
) -> Tuple[Optional[Tuple[float, float]], str]:
    """Robustly extract foot keypoint with confidence gating and fallback.

    Args:
        foot_enum: MediaPipe landmark enum for the foot index to use.
                   Defaults to LEFT_FOOT_INDEX for backward compatibility.
    """
    if foot_enum is None:
        foot_enum = mp_pose.PoseLandmark.LEFT_FOOT_INDEX

    try:
        lm = landmarks.landmark[foot_enum]
        if lm.visibility >= min_confidence:
            return (lm.x, lm.y), "primary"
    except (IndexError, AttributeError):
        pass

    try:
        lm = landmarks.landmark[foot_enum]
        if lm.visibility >= min_confidence * 0.7:
            return (lm.x, lm.y), "fallback"
    except (IndexError, AttributeError):
        pass

    return None, "missing"


def _normalize_frame_orientation(frame_bgr: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Normalize video frame orientation to ensure horizontal is true horizontal.
    Detects portrait/upside-down orientation and rotates accordingly.
    
    Note: This only rotates the frame if it's portrait. For landscape videos that are
    rotated, we detect that later based on pose orientation.
    
    Args:
        frame_bgr: Input frame in BGR format
    
    Returns:
        (normalized_frame, rotation_deg) tuple where rotation_deg is 0, 90, 180, or 270
    """
    height, width = frame_bgr.shape[:2]
    rotation_deg = 0
    
    # Detect if portrait (height > width) - rotate to landscape (90° clockwise)
    if height > width:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
        rotation_deg = 90
        print(f"Frame rotated: portrait ({height}x{width}) -> landscape (rotation: {rotation_deg}°)")
    
    # Store rotation in global state for use in angle calculations
    # Note: We may update this later based on pose auto-detection
    global _smoothing_state
    _smoothing_state["display_rotation_deg"] = rotation_deg
    
    return frame_bgr, rotation_deg


def _rotate_point_norm(p: Tuple[float, float], rot_deg: int) -> Tuple[float, float]:
    """
    Rotate a point in normalized coordinates (0..1) around center (0.5, 0.5).
    
    Args:
        p: (x, y) point in normalized coordinates
        rot_deg: Rotation in degrees (0, 90, 180, or 270)
    
    Returns:
        Rotated (x, y) point in normalized coordinates
    """
    x, y = p
    cx, cy = 0.5, 0.5
    
    # Translate to origin
    x -= cx
    y -= cy
    
    # Apply rotation
    if rot_deg == 90:
        # 90° clockwise: (x, y) -> (-y, x)
        x, y = -y, x
    elif rot_deg == 180:
        # 180°: (x, y) -> (-x, -y)
        x, y = -x, -y
    elif rot_deg == 270:
        # 270° clockwise (or -90°): (x, y) -> (y, -x)
        x, y = y, -x
    # rot_deg == 0: no change
    
    # Translate back
    x += cx
    y += cy
    
    return (x, y)


def analyze_pose_from_frame(frame_rgb: np.ndarray) -> Dict[str, object]:
    """
    Runs pose estimation on a single RGB frame and returns computed angles
    along with pixel coordinates of landmarks.
    
    Note: frame_rgb should already be in correct orientation (horizontal = true horizontal).
    """
    frame_height, frame_width = frame_rgb.shape[:2]

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        results = pose.process(frame_rgb)

    if not results.pose_landmarks:
        return {"pose_detected": False}

    landmarks = results.pose_landmarks

    # ------------------------------------------------------------------
    # 1. Select the best body side for this frame
    # ------------------------------------------------------------------
    analysis_side, side_scores = select_analysis_side(landmarks)
    side_enums = SIDE_LANDMARK_ENUMS[analysis_side]

    # ------------------------------------------------------------------
    # 2. Extract chosen-side landmarks (normalised coords)
    # ------------------------------------------------------------------
    try:
        shoulder_norm = _get_landmark_coords(landmarks, side_enums["shoulder"])
        hip_norm = _get_landmark_coords(landmarks, side_enums["hip"])
        knee_norm = _get_landmark_coords(landmarks, side_enums["knee"])
        ankle_norm = _get_landmark_coords(landmarks, side_enums["ankle"])
        elbow_norm = _get_landmark_coords(landmarks, side_enums["elbow"])
        wrist_norm = _get_landmark_coords(landmarks, side_enums["wrist"])
    except (IndexError, AttributeError):
        return {"pose_detected": False}

    foot_norm, foot_source = _robust_foot_keypoint(
        landmarks, frame_width, frame_height,
        min_confidence=0.5,
        foot_enum=side_enums["foot_index"],
    )

    # ------------------------------------------------------------------
    # 3. Pixel coordinates for drawing (chosen side)
    # ------------------------------------------------------------------
    try:
        shoulder_px = _get_landmark_pixel_coords(
            landmarks, side_enums["shoulder"], frame_width, frame_height)
        hip_px = _get_landmark_pixel_coords(
            landmarks, side_enums["hip"], frame_width, frame_height)
        knee_px = _get_landmark_pixel_coords(
            landmarks, side_enums["knee"], frame_width, frame_height)
        ankle_px = _get_landmark_pixel_coords(
            landmarks, side_enums["ankle"], frame_width, frame_height)
        elbow_px = _get_landmark_pixel_coords(
            landmarks, side_enums["elbow"], frame_width, frame_height)
        wrist_px = _get_landmark_pixel_coords(
            landmarks, side_enums["wrist"], frame_width, frame_height)
    except (IndexError, AttributeError):
        return {"pose_detected": False}

    if foot_norm is not None:
        foot_px: Optional[Tuple[int, int]] = (
            int(foot_norm[0] * frame_width),
            int(foot_norm[1] * frame_height),
        )
    else:
        try:
            foot_px = _get_landmark_pixel_coords(
                landmarks, side_enums["foot_index"], frame_width, frame_height)
        except (IndexError, AttributeError):
            foot_px = None

    # ------------------------------------------------------------------
    # 4. Reliability checks for key metrics
    # ------------------------------------------------------------------
    hip_reliable, hip_conf, hip_weak = get_metric_reliability(
        landmarks,
        [side_enums["shoulder"], side_enums["hip"], side_enums["knee"]],
    )
    torso_reliable, torso_conf, torso_weak = get_metric_reliability(
        landmarks,
        [side_enums["shoulder"], side_enums["hip"]],
    )

    # ------------------------------------------------------------------
    # 5. Compute angles — all on the chosen side, no rotation hacks
    # ------------------------------------------------------------------

    # Knee: included angle at knee (hip-knee-ankle).  Rotation-invariant.
    knee_angle = _compute_angle_deg(hip_norm, knee_norm, ankle_norm)

    # Hip: included angle at hip (shoulder-hip-knee).  Rotation-invariant.
    # This is the standard bike-fit "closed hip angle".
    hip_angle = _compute_angle_deg(shoulder_norm, hip_norm, knee_norm)

    # Elbow: included angle at elbow (shoulder-elbow-wrist).
    elbow_angle = _compute_angle_deg(shoulder_norm, elbow_norm, wrist_norm)

    if math.isnan(knee_angle) or math.isnan(hip_angle) or math.isnan(elbow_angle):
        return {"pose_detected": False}

    # Foot: optional, with EMA smoothing
    foot_angle: Optional[float] = None
    if foot_norm is not None:
        foot_angle_raw = _compute_angle_deg(knee_norm, ankle_norm, foot_norm)
        if not math.isnan(foot_angle_raw):
            foot_angle = _smooth_angle_with_ema(
                foot_angle_raw,
                _smoothing_state["foot_angle_ema"],
                alpha=0.2,
                outlier_threshold=20.0,
            )
            _smoothing_state["foot_angle_ema"] = foot_angle

    # Torso: angle of hip→shoulder line relative to real-world horizontal.
    # Uses the chosen-side shoulder and hip.  Passes the known frame
    # rotation so the vector is un-rotated back to the original camera
    # orientation before measuring the angle.
    frame_rot = _smoothing_state.get("display_rotation_deg", 0)
    torso_angle_raw, torso_diag = compute_torso_to_horizontal_deg(
        hip_norm, shoulder_norm, frame_rotation_deg=frame_rot,
    )

    torso_angle: Optional[float] = None
    if torso_angle_raw is not None:
        torso_angle = _smooth_angle_with_ema(
            torso_angle_raw,
            _smoothing_state["torso_angle_ema"],
            alpha=0.2,
            outlier_threshold=15.0,
        )
        _smoothing_state["torso_angle_ema"] = torso_angle

    _smoothing_state["torso_debug"] = {
        "analysis_side": analysis_side,
        **(torso_diag if torso_diag else {}),
    }

    # Clean optional angles
    if foot_angle is not None and math.isnan(foot_angle):
        foot_angle = None
    if torso_angle is not None and math.isnan(torso_angle):
        torso_angle = None

    # ------------------------------------------------------------------
    # 6. Collect per-landmark visibility for framing assessment
    # ------------------------------------------------------------------
    _landmark_visibility: Dict[str, Dict[str, float]] = {}
    for _name, _enum in [
        ("nose", mp_pose.PoseLandmark.NOSE),
        ("left_shoulder", mp_pose.PoseLandmark.LEFT_SHOULDER),
        ("right_shoulder", mp_pose.PoseLandmark.RIGHT_SHOULDER),
        ("left_elbow", mp_pose.PoseLandmark.LEFT_ELBOW),
        ("left_wrist", mp_pose.PoseLandmark.LEFT_WRIST),
        ("left_hip", mp_pose.PoseLandmark.LEFT_HIP),
        ("right_hip", mp_pose.PoseLandmark.RIGHT_HIP),
        ("left_knee", mp_pose.PoseLandmark.LEFT_KNEE),
        ("left_ankle", mp_pose.PoseLandmark.LEFT_ANKLE),
        ("left_foot_index", mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
    ]:
        try:
            _lm = landmarks.landmark[_enum]
            _landmark_visibility[_name] = {
                "x": _lm.x, "y": _lm.y, "confidence": _lm.visibility,
            }
        except (IndexError, AttributeError):
            _landmark_visibility[_name] = {"x": 0.0, "y": 0.0, "confidence": 0.0}

    # ------------------------------------------------------------------
    # 7. Build result dict (backward-compatible keys)
    # ------------------------------------------------------------------
    result: Dict[str, object] = {
        "pose_detected": True,
        "analysis_side": analysis_side,
        "knee_angle_deg": round(knee_angle, 2),
        "hip_angle_deg": round(hip_angle, 2),
        "elbow_angle_deg": round(elbow_angle, 2),
        "foot_angle_deg": round(foot_angle, 2) if foot_angle is not None else None,
        "torso_angle_deg": round(torso_angle, 2) if torso_angle is not None else None,
        "torso_debug": _smoothing_state.get("torso_debug"),
        "landmarks_px": {
            "shoulder": shoulder_px,
            "hip": hip_px,
            "knee": knee_px,
            "ankle": ankle_px,
            "elbow": elbow_px,
            "wrist": wrist_px,
        },
        "landmark_visibility": _landmark_visibility,
        "angle_reliability": {
            "hip": {"reliable": hip_reliable, "confidence": round(hip_conf, 3), "weak": hip_weak},
            "torso": {"reliable": torso_reliable, "confidence": round(torso_conf, 3), "weak": torso_weak},
            "side_scores": {k: round(v, 3) for k, v in side_scores.items()},
        },
    }

    if foot_px is not None:
        result["landmarks_px"]["foot"] = foot_px

    return result


def draw_pose_overlay(
    frame: np.ndarray,
    landmarks_px: Dict[str, Tuple[int, int]],
    knee_angle_deg: float,
    hip_angle_deg: float,
    foot_angle_deg: Optional[float] = None,
    torso_angle_deg: Optional[float] = None,
    elbow_angle_deg: Optional[float] = None,
) -> np.ndarray:
    """
    Draws pose overlay on a frame with joint markers, connecting lines, and angle labels.
    Returns a copy of the frame with annotations.
    """
    annotated = frame.copy()

    shoulder = landmarks_px["shoulder"]
    hip = landmarks_px["hip"]
    knee = landmarks_px["knee"]
    ankle = landmarks_px["ankle"]
    foot = landmarks_px.get("foot")

    # Color scheme: joints in blue, lines in green, text in white
    joint_color = (0, 165, 255)  # Orange in BGR
    line_color = (0, 255, 0)  # Green in BGR
    text_color = (255, 255, 255)  # White in BGR
    text_bg_color = (0, 0, 0)  # Black background for text

    # Draw connecting lines: shoulder -> hip -> knee -> ankle -> foot
    cv2.line(annotated, shoulder, hip, line_color, 3)
    cv2.line(annotated, hip, knee, line_color, 3)
    cv2.line(annotated, knee, ankle, line_color, 3)
    if foot:
        cv2.line(annotated, ankle, foot, line_color, 3)
    
    # Draw arm lines if elbow and wrist are available
    if "elbow" in landmarks_px and "wrist" in landmarks_px:
        elbow = landmarks_px["elbow"]
        cv2.line(annotated, shoulder, elbow, line_color, 3)

    # Draw joint markers (circles)
    joint_radius = 8
    cv2.circle(annotated, shoulder, joint_radius, joint_color, -1)
    cv2.circle(annotated, hip, joint_radius, joint_color, -1)
    cv2.circle(annotated, knee, joint_radius, joint_color, -1)
    cv2.circle(annotated, ankle, joint_radius, joint_color, -1)
    if foot:
        cv2.circle(annotated, foot, joint_radius, joint_color, -1)
    if "elbow" in landmarks_px:
        elbow = landmarks_px["elbow"]
        cv2.circle(annotated, elbow, joint_radius, joint_color, -1)

    # Draw angle labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Knee angle label (near knee joint)
    knee_text = f"Knee: {knee_angle_deg:.1f}°"
    knee_text_size = cv2.getTextSize(knee_text, font, font_scale, thickness)[0]
    knee_text_x = knee[0] + 15
    knee_text_y = knee[1] - 10

    # Draw text background rectangle
    cv2.rectangle(
        annotated,
        (knee_text_x - 5, knee_text_y - knee_text_size[1] - 5),
        (knee_text_x + knee_text_size[0] + 5, knee_text_y + 5),
        text_bg_color,
        -1,
    )
    cv2.putText(annotated, knee_text, (knee_text_x, knee_text_y), font, font_scale, text_color, thickness)

    # Hip angle label (near hip joint)
    hip_text = f"Hip: {hip_angle_deg:.1f}°"
    hip_text_size = cv2.getTextSize(hip_text, font, font_scale, thickness)[0]
    hip_text_x = hip[0] + 15
    hip_text_y = hip[1] - 10

    # Draw text background rectangle
    cv2.rectangle(
        annotated,
        (hip_text_x - 5, hip_text_y - hip_text_size[1] - 5),
        (hip_text_x + hip_text_size[0] + 5, hip_text_y + 5),
        text_bg_color,
        -1,
    )
    cv2.putText(annotated, hip_text, (hip_text_x, hip_text_y), font, font_scale, text_color, thickness)

    # Foot angle label (near ankle/foot)
    if "foot" in landmarks_px and foot_angle_deg is not None:
        foot = landmarks_px["foot"]
        ankle = landmarks_px["ankle"]
        foot_text = f"Foot: {foot_angle_deg:.1f}°"
        foot_text_size = cv2.getTextSize(foot_text, font, font_scale, thickness)[0]
        foot_text_x = ankle[0] - foot_text_size[0] - 10
        foot_text_y = ankle[1] + foot_text_size[1] + 10

        cv2.rectangle(
            annotated,
            (foot_text_x - 5, foot_text_y - foot_text_size[1] - 5),
            (foot_text_x + foot_text_size[0] + 5, foot_text_y + 5),
            text_bg_color,
            -1,
        )
        cv2.putText(annotated, foot_text, (foot_text_x, foot_text_y), font, font_scale, text_color, thickness)

    # Torso angle label (using chosen-side shoulder/hip)
    if torso_angle_deg is not None:
        torso_mid_x = (hip[0] + shoulder[0]) // 2
        torso_mid_y = (hip[1] + shoulder[1]) // 2

        torso_text = f"Torso (to horizontal): {torso_angle_deg:.1f}\u00b0"
        torso_text_size = cv2.getTextSize(torso_text, font, font_scale, thickness)[0]
        torso_text_x = torso_mid_x - torso_text_size[0] // 2
        torso_text_y = torso_mid_y - 20

        cv2.rectangle(
            annotated,
            (torso_text_x - 5, torso_text_y - torso_text_size[1] - 5),
            (torso_text_x + torso_text_size[0] + 5, torso_text_y + 5),
            text_bg_color,
            -1,
        )
        cv2.putText(annotated, torso_text, (torso_text_x, torso_text_y), font, font_scale, text_color, thickness)

    # Elbow angle label (near elbow joint)
    if "elbow" in landmarks_px and "wrist" in landmarks_px and elbow_angle_deg is not None:
        elbow = landmarks_px["elbow"]
        elbow_text = f"Elbow: {elbow_angle_deg:.1f}°"
        elbow_text_size = cv2.getTextSize(elbow_text, font, font_scale, thickness)[0]
        elbow_text_x = elbow[0] + 15
        elbow_text_y = elbow[1] - 10

        cv2.rectangle(
            annotated,
            (elbow_text_x - 5, elbow_text_y - elbow_text_size[1] - 5),
            (elbow_text_x + elbow_text_size[0] + 5, elbow_text_y + 5),
            text_bg_color,
            -1,
        )
        cv2.putText(annotated, elbow_text, (elbow_text_x, elbow_text_y), font, font_scale, text_color, thickness)

        # Draw elbow-wrist line
        wrist = landmarks_px["wrist"]
        cv2.line(annotated, elbow, wrist, line_color, 2)
        cv2.circle(annotated, wrist, joint_radius, joint_color, -1)

    return annotated


def generate_bikefit_recommendations(
    angles: Dict[str, Optional[float]],
    bike_type: str = "road",
    bike_config: Optional[dict] = None,
    goal: str = "Balanced",
    mobility: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """
    Evidence-based recommendations with bike-type-specific thresholds.

    angles keys:
        knee_angle_deg (float, required)
        hip_angle_deg (float, required)
        foot_angle_deg (float | None, optional)
        torso_angle_deg (float | None, optional)
    """
    if bike_config is None:
        # Fallback to road defaults if config not provided
        bike_config = {
            "knee": {"optimal": (138.0, 145.0), "neutral": (135.0, 148.0)},
            "hip": {"optimal": (100.0, 115.0), "neutral": (95.0, 120.0)},
            "foot": {"neutral": (85.0, 95.0), "ok": (82.0, 98.0)},
        }

    knee = angles.get("knee_angle_deg")
    hip = angles.get("hip_angle_deg")
    foot = angles.get("foot_angle_deg")
    torso = angles.get("torso_angle_deg")
    elbow = angles.get("elbow_angle_deg")

    def fmt(v: Optional[float]) -> str:
        return f"{v:.1f}°" if v is not None else "n/a"

    knee_cfg = bike_config["knee"]
    hip_cfg = bike_config["hip"]
    foot_cfg = bike_config["foot"]

    # Knee logic
    if knee is None:
        knee_comment = "Knee angle not available."
    else:
        opt_min, opt_max = knee_cfg["optimal"]
        neut_min, neut_max = knee_cfg["neutral"]
        if opt_min <= knee <= opt_max:
            knee_comment = f"Knee angle is within the optimal range for this bike type. Measured: {fmt(knee)}"
        elif neut_min <= knee < opt_min:
            knee_comment = (
                f"Knee angle is acceptable but could be refined. Measured: {fmt(knee)}. "
                f"Small changes to saddle height (±3–5 mm) could fine-tune comfort and power."
            )
        elif opt_max < knee <= neut_max:
            knee_comment = (
                f"Knee angle is acceptable but could be refined. Measured: {fmt(knee)}. "
                f"Small changes to saddle height (±3–5 mm) could fine-tune comfort and power."
            )
        elif knee < neut_min:
            knee_comment = (
                f"Knee angle is too small (leg too flexed). This usually means the saddle is too low. "
                f"Consider raising the saddle ~5–10 mm and re-testing. Measured: {fmt(knee)}"
            )
        else:  # knee > neut_max
            knee_comment = (
                f"Knee angle is too large (leg very straight). This can increase strain behind the knee. "
                f"Consider lowering the saddle slightly or moving it forward a few mm and re-testing. Measured: {fmt(knee)}"
            )

    # Hip logic
    if hip is None:
        hip_comment = "Hip angle not available."
    else:
        opt_min, opt_max = hip_cfg["optimal"]
        neut_min, neut_max = hip_cfg["neutral"]
        if opt_min <= hip <= opt_max:
            hip_comment = (
                f"Hip angle is in a strong range for this bike type – good balance between power and aerodynamics. "
                f"Measured: {fmt(hip)}"
            )
        elif neut_min <= hip < opt_min:
            hip_comment = (
                f"Hip angle is acceptable but could be refined. Measured: {fmt(hip)}. "
                f"Consider small adjustments to bar height or saddle position."
            )
        elif opt_max < hip <= neut_max:
            hip_comment = (
                f"Hip angle is acceptable but could be refined. Measured: {fmt(hip)}. "
                f"Consider small adjustments to bar height or saddle position."
            )
        elif hip < neut_min:
            if bike_type == "tt":
                hip_comment = (
                    f"Hip angle is very closed for a TT position. This can restrict power and stress the lower back. "
                    f"Consider raising the front end slightly, shortening the reach, or using shorter cranks to open the hip. "
                    f"Measured: {fmt(hip)}"
                )
            else:
                hip_comment = (
                    f"Hip angle is very closed for this bike type. Try raising the bars or moving the saddle slightly back/up "
                    f"to open the hip angle. Measured: {fmt(hip)}"
                )
        else:  # hip > neut_max
            if bike_type == "tt":
                hip_comment = (
                    f"Hip angle is very open. This is comfortable but may give away aerodynamics. "
                    f"If comfort allows, lowering the front end slightly could reduce drag. Measured: {fmt(hip)}"
                )
            else:
                hip_comment = (
                    f"Hip angle is quite open. If you feel too upright or overloaded on the saddle, "
                    f"consider a small bar drop or moving the saddle slightly forward. Measured: {fmt(hip)}"
                )

    # Foot logic
    if foot is None:
        foot_comment = "Foot angle not measured."
    else:
        neut_min, neut_max = foot_cfg["neutral"]
        ok_min, ok_max = foot_cfg["ok"]
        if neut_min <= foot <= neut_max:
            foot_comment = f"Foot angle is in a strong neutral range ({neut_min:.0f}–{neut_max:.0f}°). Measured: {fmt(foot)}"
        elif ok_min <= foot < neut_min:
            foot_comment = (
                f"Foot angle is acceptable but slightly heel-down. Measured: {fmt(foot)}. "
                f"If you have calf fatigue or hot spots, experimenting with cleat fore-aft or saddle height may help."
            )
        elif neut_max < foot <= ok_max:
            foot_comment = (
                f"Foot angle is acceptable but slightly toes-down. Measured: {fmt(foot)}. "
                f"If you have calf fatigue or hot spots, experimenting with cleat fore-aft or saddle height may help."
            )
        elif foot < ok_min:
            foot_comment = (
                f"Foot is quite heel-down at the measured frame. This can increase ankle work and may indicate "
                f"the saddle is slightly high or too far back. Measured: {fmt(foot)}"
            )
        else:  # foot > ok_max
            foot_comment = (
                f"Foot is quite toes-down at the measured frame. This can overload the calf and Achilles. "
                f"Check that the saddle isn't too low or too far forward, and that cleats aren't excessively forward. "
                f"Measured: {fmt(foot)}"
            )

    # Torso logic (bike-type-specific)
    if torso is None:
        torso_comment = "Torso angle not measured."
    else:
        if bike_type == "tt":
            if torso < 10:
                torso_comment = (
                    f"Torso extremely low (<10°). Very aero but likely unsustainable "
                    f"and may close hip angle too much. Measured: {fmt(torso)}"
                )
            elif 10 <= torso <= 20:
                torso_comment = (
                    f"Aggressive aero torso angle (10–20°). Good for short TTs if "
                    f"you can sustain power and comfort. Measured: {fmt(torso)}"
                )
            elif 20 < torso <= 30:
                torso_comment = (
                    f"Balanced aero/comfort torso angle (20–30°). Typical for long-course "
                    f"triathlon or sustainable TT positions. Measured: {fmt(torso)}"
                )
            else:  # torso > 30
                torso_comment = (
                    f"Torso relatively upright (>30°). Comfortable but giving up aero "
                    f"benefits; consider lowering front end or extending reach. Measured: {fmt(torso)}"
                )
        elif bike_type == "road":
            if torso < 30:
                torso_comment = (
                    f"Torso very low (<30°). Very aggressive position; may be hard to sustain. "
                    f"Consider raising bars slightly if comfort is an issue. Measured: {fmt(torso)}"
                )
            elif 30 <= torso <= 40:
                torso_comment = (
                    f"Performance-oriented torso angle (30–40°). Good balance of aero and power. "
                    f"Measured: {fmt(torso)}"
                )
            elif 40 < torso <= 50:
                torso_comment = (
                    f"Endurance/comfort torso angle (40–50°). Good for long rides and sustainable power. "
                    f"Measured: {fmt(torso)}"
                )
            else:  # torso > 50
                torso_comment = (
                    f"Torso relatively upright (>50°). Comfortable but less aero; consider lowering bars "
                    f"slightly if you want more performance. Measured: {fmt(torso)}"
                )
        elif bike_type == "gravel":
            if torso < 40:
                torso_comment = (
                    f"Torso quite low (<40°). May be too aggressive for gravel riding; consider raising bars. "
                    f"Measured: {fmt(torso)}"
                )
            elif 40 <= torso <= 55:
                torso_comment = (
                    f"Stable handling & long-ride comfort torso angle (40–55°). Good for mixed terrain. "
                    f"Measured: {fmt(torso)}"
                )
            else:  # torso > 55
                torso_comment = (
                    f"Torso quite upright (>55°). Very comfortable but may limit control on descents; "
                    f"consider slight bar drop. Measured: {fmt(torso)}"
                )
        else:  # mtb
            if torso < 45:
                torso_comment = (
                    f"Torso quite low (<45°). May be too aggressive for technical MTB; consider raising bars. "
                    f"Measured: {fmt(torso)}"
                )
            elif 45 <= torso <= 60:
                torso_comment = (
                    f"Technical control & shock absorption torso angle (45–60°). Good for variable terrain. "
                    f"Measured: {fmt(torso)}"
                )
            else:  # torso > 60
                torso_comment = (
                    f"Torso very upright (>60°). Very comfortable but may limit control; "
                    f"consider slight bar adjustment. Measured: {fmt(torso)}"
                )

    # Elbow logic (bike-type-specific)
    if elbow is None:
        elbow_comment = "Elbow angle not measured."
    else:
        if bike_type == "tt":
            if elbow < 90:
                elbow_comment = (
                    f"Elbows very closed (<90°). Can feel cramped and restrict breathing. "
                    f"Measured: {fmt(elbow)}"
                )
            elif 90 <= elbow <= 105:
                elbow_comment = (
                    f"Elbow angle in an aero range (90–105°). Compact and aerodynamic. "
                    f"Measured: {fmt(elbow)}"
                )
            elif 105 < elbow <= 115:
                elbow_comment = (
                    f"Elbow angle in a comfort-aero range (105–115°). A good compromise "
                    f"for longer events. Measured: {fmt(elbow)}"
                )
            else:  # elbow > 115
                elbow_comment = (
                    f"Elbows quite open (>115°). More comfortable but less aero; consider "
                    f"bringing pads back or narrowing reach if you want more aero. Measured: {fmt(elbow)}"
                )
        elif bike_type == "road":
            if elbow < 130:
                elbow_comment = (
                    f"Elbows very tight (<130°). Very tight bend can cause shoulder/neck fatigue. "
                    f"Consider adjusting bar height or reach. Measured: {fmt(elbow)}"
                )
            elif 130 <= elbow < 140:
                elbow_comment = (
                    f"Elbow angle slightly tight (130–140°). May cause fatigue on long rides. "
                    f"Measured: {fmt(elbow)}"
                )
            elif 140 <= elbow <= 165:
                elbow_comment = (
                    f"Neutral/performance elbow angle (140–165°). Relaxed bend, good for comfort "
                    f"and shock absorption. Measured: {fmt(elbow)}"
                )
            elif 165 < elbow <= 170:
                elbow_comment = (
                    f"Elbow angle slightly open (165–170°). Still acceptable but approaching locked position. "
                    f"Measured: {fmt(elbow)}"
                )
            else:  # elbow > 170
                elbow_comment = (
                    f"Elbows almost locked (>170°). Poor shock absorption; consider slight bend "
                    f"for better comfort and control. Measured: {fmt(elbow)}"
                )
        else:  # gravel or mtb
            if elbow < 135:
                elbow_comment = (
                    f"Elbows very tight (<135°). Can cause fatigue; consider adjusting bar position. "
                    f"Measured: {fmt(elbow)}"
                )
            elif 135 <= elbow < 145:
                elbow_comment = (
                    f"Elbow angle slightly tight (135–145°). May cause fatigue on long rides. "
                    f"Measured: {fmt(elbow)}"
                )
            elif 145 <= elbow <= 170:
                elbow_comment = (
                    f"Neutral elbow angle (145–170°). Slightly bent for comfort and control. "
                    f"Measured: {fmt(elbow)}"
                )
            else:  # elbow > 170
                elbow_comment = (
                    f"Elbows nearly locked (>170°). Poor shock absorption; consider slight bend "
                    f"for better control on rough terrain. Measured: {fmt(elbow)}"
                )

    # Summary paragraph
    summary_parts = []
    if knee is not None:
        opt_min, opt_max = knee_cfg["optimal"]
        neut_min, neut_max = knee_cfg["neutral"]
        if knee < neut_min:
            summary_parts.append("Knee likely too flexed; saddle may be too low.")
        elif knee > neut_max:
            summary_parts.append("Knee likely too extended; saddle may be too high.")
        elif opt_min <= knee <= opt_max:
            summary_parts.append("Knee angle optimal.")
        else:
            summary_parts.append("Knee angle acceptable but could be refined.")

    if hip is not None:
        opt_min, opt_max = hip_cfg["optimal"]
        neut_min, neut_max = hip_cfg["neutral"]
        if hip < neut_min:
            summary_parts.append("Hip very closed; may restrict power.")
        elif hip > neut_max:
            summary_parts.append("Hip quite open; may give away aero or comfort.")
        elif opt_min <= hip <= opt_max:
            summary_parts.append("Hip angle strong.")
        else:
            summary_parts.append("Hip angle acceptable.")

    if foot is not None:
        neut_min, neut_max = foot_cfg["neutral"]
        ok_min, ok_max = foot_cfg["ok"]
        if foot < ok_min or foot > ok_max:
            summary_parts.append("Foot angle outside ideal range.")
        elif neut_min <= foot <= neut_max:
            summary_parts.append("Foot angle neutral.")
        else:
            summary_parts.append("Foot angle acceptable.")

    if torso is not None:
        if bike_type == "tt":
            if torso < 10:
                summary_parts.append("Torso extremely low; may be unsustainable.")
            elif 10 <= torso <= 30:
                summary_parts.append("Torso angle good for aero.")
            else:
                summary_parts.append("Torso relatively upright for TT; consider lowering front end.")
        elif bike_type == "road":
            if 30 <= torso <= 50:
                summary_parts.append("Torso angle appropriate for road.")
            elif torso < 30:
                summary_parts.append("Torso very aggressive; may be hard to sustain.")
            else:
                summary_parts.append("Torso relatively upright; consider lowering bars.")
        elif bike_type == "gravel":
            if 40 <= torso <= 55:
                summary_parts.append("Torso angle good for gravel.")
            else:
                summary_parts.append("Torso angle may need adjustment for gravel.")
        else:  # mtb
            if 45 <= torso <= 60:
                summary_parts.append("Torso angle good for MTB.")
            else:
                summary_parts.append("Torso angle may need adjustment for MTB.")

    if elbow is not None:
        if bike_type == "tt":
            if 90 <= elbow <= 115:
                summary_parts.append("Elbow angle good for aero.")
            elif elbow < 90:
                summary_parts.append("Elbows very closed; may restrict breathing.")
            else:
                summary_parts.append("Elbows quite open; consider bringing pads back for more aero.")
        elif bike_type == "road":
            if 140 <= elbow <= 165:
                summary_parts.append("Elbow angle good for road.")
            elif elbow < 130:
                summary_parts.append("Elbows very tight; may cause fatigue.")
            else:
                summary_parts.append("Elbow angle acceptable but could be refined.")
        else:  # gravel or mtb
            if 145 <= elbow <= 170:
                summary_parts.append("Elbow angle good for control.")
            elif elbow < 135:
                summary_parts.append("Elbows very tight; may cause fatigue.")
            else:
                summary_parts.append("Elbow angle acceptable but could be refined.")

    summary = " ".join(summary_parts)
    if bike_config.get("summary_focus"):
        summary += f" {bike_config['summary_focus']}"

    return {
        "knee_comment": knee_comment,
        "hip_comment": hip_comment,
        "foot_comment": foot_comment,
        "torso_comment": torso_comment,
        "elbow_comment": elbow_comment,
        "summary": summary,
    }


def generate_report(
    knee_angle_deg: float, hip_angle_deg: float, recommendations: Dict[str, str], bike_config: Optional[dict] = None
) -> str:
    """
    Generates a concise human-friendly summary using the recommendation summary.
    """
    bike_label = bike_config.get("label", "Road bike") if bike_config else "Road bike"
    base_summary = recommendations.get(
        "summary",
        f"Knee angle: {knee_angle_deg:.1f}°. Hip angle: {hip_angle_deg:.1f}°.",
    )
    return f"Bike type: {bike_label}. {base_summary}"


def get_target_ranges(bike_type: str, goal: str, mobility: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """
    Get target ranges for each angle based on bike type, goal, and mobility.
    Goal sensitivity: Comfort = wider range, Aero-Performance = narrower range.
    Mobility adjusts ranges slightly (soft weighting).
    """
    # Base ranges by bike type (from BIKE_TYPE_CONFIG)
    base_ranges = {
        "tt": {
            "knee": (138.0, 152.0),
            "hip": (90.0, 115.0),
            "foot": (82.0, 98.0),
            "torso": (10.0, 30.0),
            "elbow": (90.0, 115.0),
        },
        "road": {
            "knee": (135.0, 148.0),
            "hip": (95.0, 120.0),
            "foot": (82.0, 98.0),
            "torso": (30.0, 50.0),
            "elbow": (140.0, 165.0),
        },
        "gravel": {
            "knee": (135.0, 148.0),
            "hip": (100.0, 125.0),
            "foot": (82.0, 98.0),
            "torso": (40.0, 55.0),
            "elbow": (145.0, 170.0),
        },
        "mtb": {
            "knee": (135.0, 148.0),
            "hip": (105.0, 130.0),
            "foot": (82.0, 98.0),
            "torso": (45.0, 60.0),
            "elbow": (145.0, 170.0),
        },
    }

    ranges = base_ranges.get(bike_type, base_ranges["road"]).copy()

    # Apply goal sensitivity
    if goal == "Comfort":
        # Widen ranges by 5-10%
        for key in ranges:
            min_val, max_val = ranges[key]
            width = max_val - min_val
            ranges[key] = (min_val - width * 0.1, max_val + width * 0.1)
    elif goal == "Aero-Performance":
        # Narrow ranges by 5-10%, shift toward aggressive end
        for key in ranges:
            min_val, max_val = ranges[key]
            width = max_val - min_val
            if key in ["torso", "hip"]:  # Lower is better for aero
                ranges[key] = (min_val, max_val - width * 0.1)
            else:
                ranges[key] = (min_val + width * 0.05, max_val - width * 0.05)

    # Soft mobility adjustments (if mobility scores are low, widen ranges slightly)
    avg_mobility = sum(mobility.values()) / len(mobility) if mobility else 7.0
    if avg_mobility < 5.0:  # Low mobility
        for key in ranges:
            min_val, max_val = ranges[key]
            width = max_val - min_val
            ranges[key] = (min_val - width * 0.05, max_val + width * 0.05)

    return ranges


def compute_fit_windows(
    angles: Dict[str, Optional[float]], target_ranges: Dict[str, Tuple[float, float]]
) -> Dict[str, Dict[str, any]]:
    """
    Compute fit windows for each angle: measured value, target range, and status.
    Status: "In Range", "Slightly Off", "Off"
    """
    fit_windows = {}

    for angle_name, measured in angles.items():
        if measured is None:
            continue

        # Extract metric name (e.g., "knee_angle_deg" -> "knee")
        metric = angle_name.replace("_angle_deg", "")
        if metric not in target_ranges:
            continue

        target_min, target_max = target_ranges[metric]

        # Determine status
        if target_min <= measured <= target_max:
            status = "In Range"
        else:
            # Calculate how far off
            if measured < target_min:
                diff = target_min - measured
            else:
                diff = measured - target_max

            # Percentage of range width
            range_width = target_max - target_min
            if range_width > 0:
                off_percentage = (diff / range_width) * 100
                if off_percentage < 20:
                    status = "Slightly Off"
                else:
                    status = "Off"
            else:
                status = "Off"

        fit_windows[metric] = {
            "measured": round(measured, 1),
            "target_min": round(target_min, 1),
            "target_max": round(target_max, 1),
            "status": status,
        }

    return fit_windows


def compute_stroke_samples(
    frame_results: List[Tuple[int, any, Dict, float]], min_knee_idx: int, max_knee_idx: int, total_frames: int
) -> Dict[str, Dict[str, float]]:
    """
    Compute stroke samples: top (max knee = bottom of stroke), bottom (min knee = top of stroke), mid.
    Returns angles at each position.
    """
    # Find frames at each position
    top_frame = None  # Max knee angle = bottom of stroke
    bottom_frame = None  # Min knee angle = top of stroke
    mid_frame = None

    for idx, frame_bgr, pose_result, knee_angle in frame_results:
        if idx == max_knee_idx:
            top_frame = pose_result
        elif idx == min_knee_idx:
            bottom_frame = pose_result

    # Find mid frame (average index)
    mid_idx = (min_knee_idx + max_knee_idx) // 2
    for idx, frame_bgr, pose_result, knee_angle in frame_results:
        if abs(idx - mid_idx) < 5:  # Within 5 frames of mid
            mid_frame = pose_result
            break

    def extract_angles(pose_result: Optional[Dict]) -> Dict[str, Optional[float]]:
        if pose_result is None:
            return {
                "knee_angle_deg": None,
                "hip_angle_deg": None,
                "foot_angle_deg": None,
                "torso_angle_deg": None,
                "elbow_angle_deg": None,
            }
        return {
            "knee_angle_deg": pose_result.get("knee_angle_deg"),
            "hip_angle_deg": pose_result.get("hip_angle_deg"),
            "foot_angle_deg": pose_result.get("foot_angle_deg"),
            "torso_angle_deg": pose_result.get("torso_angle_deg"),
            "elbow_angle_deg": pose_result.get("elbow_angle_deg"),
        }

    return {
        "top": extract_angles(top_frame),  # Bottom of stroke (leg extended)
        "mid": extract_angles(mid_frame),
        "bottom": extract_angles(bottom_frame),  # Top of stroke (leg flexed)
    }


def generate_recommended_actions(
    fit_windows: Dict[str, Dict[str, any]], bike_type: str, goal: str
) -> List[Dict[str, str]]:
    """
    Generate prioritized recommended actions based on fit windows.
    Priority: pain-risk metrics first, then aero/comfort.
    Returns max 3 actions.
    """
    actions = []

    # Priority order: knee extremes, hip extremes, then others
    priority_order = ["knee", "hip", "foot", "torso", "elbow"]

    for metric in priority_order:
        if metric not in fit_windows:
            continue

        window = fit_windows[metric]
        if window["status"] == "In Range":
            continue

        measured = window["measured"]
        target_min = window["target_min"]
        target_max = window["target_max"]

        # Determine adjustment
        if measured < target_min:
            diff = target_min - measured
            direction = "increase"
        else:
            diff = measured - target_max
            direction = "decrease"

        # Generate action based on metric
        if metric == "knee":
            if direction == "increase":
                action = {
                    "title": "Raise Saddle",
                    "change": f"{int(diff * 2.5)}-{int(diff * 3.5)} mm",
                    "reason": f"Knee angle {measured:.1f}° is below target range ({target_min:.1f}–{target_max:.1f}°). Saddle too low.",
                    "priority": 1 if diff > 5 else 2,
                }
            else:
                action = {
                    "title": "Lower Saddle",
                    "change": f"{int(diff * 2.5)}-{int(diff * 3.5)} mm",
                    "reason": f"Knee angle {measured:.1f}° is above target range ({target_min:.1f}–{target_max:.1f}°). Saddle too high.",
                    "priority": 1 if diff > 5 else 2,
                }
        elif metric == "hip":
            if direction == "increase":
                action = {
                    "title": "Open Hip Angle",
                    "change": "Raise bars 10–20 mm or slide saddle back 5–10 mm",
                    "reason": f"Hip angle {measured:.1f}° is below target range ({target_min:.1f}–{target_max:.1f}°). Too closed.",
                    "priority": 2,
                }
            else:
                action = {
                    "title": "Close Hip Angle",
                    "change": "Lower bars 10–20 mm or slide saddle forward 5–10 mm",
                    "reason": f"Hip angle {measured:.1f}° is above target range ({target_min:.1f}–{target_max:.1f}°). Too open.",
                    "priority": 3,
                }
        elif metric == "foot":
            if direction == "increase":
                action = {
                    "title": "Adjust Foot Position",
                    "change": "Lower saddle 3–8 mm or adjust cleat position",
                    "reason": f"Foot angle {measured:.1f}° is below target range ({target_min:.1f}–{target_max:.1f}°).",
                    "priority": 3,
                }
            else:
                action = {
                    "title": "Adjust Foot Position",
                    "change": "Raise saddle 3–8 mm or adjust cleat position",
                    "reason": f"Foot angle {measured:.1f}° is above target range ({target_min:.1f}–{target_max:.1f}°).",
                    "priority": 3,
                }
        else:
            # Torso or elbow - lower priority
            action = {
                "title": f"Adjust {metric.capitalize()} Position",
                "change": "Adjust bar height or reach",
                "reason": f"{metric.capitalize()} angle {measured:.1f}° is outside target range ({target_min:.1f}–{target_max:.1f}°).",
                "priority": 3,
            }

        actions.append(action)

    # Sort by priority and return top 3
    actions.sort(key=lambda x: x["priority"])
    return actions[:3]


def generate_annotated_video(
    input_video_path: str,
    output_video_path: str,
    sample_every_n_frames: int = 1,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Reads the input video frame by frame, runs pose detection,
    draws the overlay (joints + lines + angles) on each frame,
    and writes an annotated video to output_video_path.

    Args:
        input_video_path: Path to input video file
        output_video_path: Path where annotated video will be saved
        sample_every_n_frames: Process every Nth frame (1 = all frames)

    Returns:
        (avg_knee_angle_deg, avg_hip_angle_deg) over all frames where pose was detected,
        or (None, None) if no poses detected
    """
    # Reset smoothing state for new video
    global _smoothing_state
    _smoothing_state["foot_angle_ema"] = None
    _smoothing_state["torso_angle_ema"] = None
    _smoothing_state["torso_debug"] = None
    _smoothing_state["display_rotation_deg"] = 0  # Reset rotation tracking

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # For portrait videos, the output will be landscape after normalization
    if height > width:
        out_width, out_height = height, width
    else:
        out_width, out_height = width, height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    # Initialize MediaPipe pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,  # Use video mode for better performance
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    knee_angles = []
    hip_angles = []
    foot_angles = []
    frame_idx = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n_frames == 0:
                # Normalize orientation (portrait → landscape) the same way
                # the main analysis pipeline does, so torso angle compensation
                # uses the correct frame_rotation_deg.
                frame_bgr_normalized, _ = _normalize_frame_orientation(frame_bgr.copy())
                frame_rgb = cv2.cvtColor(frame_bgr_normalized, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    pose_result = analyze_pose_from_frame(frame_rgb)
                    if pose_result.get("pose_detected"):
                        knee_angle = pose_result["knee_angle_deg"]
                        hip_angle = pose_result["hip_angle_deg"]
                        foot_angle = pose_result.get("foot_angle_deg")
                        torso_angle = pose_result.get("torso_angle_deg")
                        elbow_angle = pose_result.get("elbow_angle_deg")
                        landmarks_px = pose_result["landmarks_px"]

                        knee_angles.append(knee_angle)
                        hip_angles.append(hip_angle)
                        if foot_angle is not None:
                            foot_angles.append(foot_angle)

                        annotated_frame = draw_pose_overlay(
                            frame_rgb,
                            landmarks_px,
                            knee_angle,
                            hip_angle,
                            foot_angle_deg=foot_angle,
                            torso_angle_deg=torso_angle,
                            elbow_angle_deg=elbow_angle,
                        )
                        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                        out.write(annotated_frame_bgr)
                    else:
                        out.write(frame_bgr_normalized)
                else:
                    out.write(frame_bgr_normalized)
            else:
                # For non-processed frames, still normalize orientation
                frame_bgr_normalized, _ = _normalize_frame_orientation(frame_bgr.copy())
                out.write(frame_bgr_normalized)

            frame_idx += 1

    finally:
        cap.release()
        out.release()
        pose.close()

    # Calculate averages
    avg_knee = sum(knee_angles) / len(knee_angles) if knee_angles else None
    avg_hip = sum(hip_angles) / len(hip_angles) if hip_angles else None
    avg_foot = sum(foot_angles) / len(foot_angles) if foot_angles else None

    return (avg_knee, avg_hip, avg_foot)

