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
    "torso_debug": None,  # Store dx, dy for debugging
    "display_rotation_deg": 0,  # Rotation applied to frames (0, 90, 180, 270)
}


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
    landmarks: mp.solutions.pose.PoseLandmark,
    frame_width: int,
    frame_height: int,
    min_confidence: float = 0.5,
) -> Tuple[Optional[Tuple[float, float]], str]:
    """
    Robustly extract foot keypoint with confidence gating and fallback strategy.
    
    Args:
        landmarks: MediaPipe pose landmarks
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        min_confidence: Minimum confidence threshold
    
    Returns:
        (foot_point, source) where foot_point is (x, y) normalized coords or None,
        and source is "primary", "fallback", or "missing"
    """
    # Primary: foot_index (toe)
    try:
        foot_index_landmark = landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        foot_index_conf = foot_index_landmark.visibility  # MediaPipe uses visibility as confidence
        
        if foot_index_conf >= min_confidence:
            foot_index_norm = (foot_index_landmark.x, foot_index_landmark.y)
            return foot_index_norm, "primary"
    except (IndexError, AttributeError):
        pass
    
    # Fallback: heel (if available) or foot_index with lower confidence
    try:
        # Try heel if available (some MediaPipe models have it)
        # For now, fallback to foot_index with lower threshold
        foot_index_landmark = landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        foot_index_conf = foot_index_landmark.visibility
        
        if foot_index_conf >= min_confidence * 0.7:  # Lower threshold for fallback
            foot_index_norm = (foot_index_landmark.x, foot_index_landmark.y)
            return foot_index_norm, "fallback"
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
    try:
        # Get mid-torso points for stable measurement
        mid_shoulder_norm, mid_hip_norm = _get_mid_torso_points(landmarks)
        
        if mid_shoulder_norm is None or mid_hip_norm is None:
            # Fallback to left side if mid-points unavailable
            try:
                mid_shoulder_norm = _get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
                mid_hip_norm = _get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
            except (IndexError, AttributeError):
                return {"pose_detected": False}
        
        # Get other landmarks (always use left side for now, can be extended)
        left_shoulder_norm = _get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        left_hip_norm = _get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        knee_norm = _get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
        ankle_norm = _get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        elbow_norm = _get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
        wrist_norm = _get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
        
        # Robust foot keypoint detection with confidence gating and fallback
        foot_norm, foot_source = _robust_foot_keypoint(landmarks, frame_width, frame_height, min_confidence=0.5)

        # Pixel coordinates for drawing
        # Get mid-torso points for visualization
        left_shoulder_px = _get_landmark_pixel_coords(
            landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, frame_width, frame_height
        )
        right_shoulder_px = _get_landmark_pixel_coords(
            landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, frame_width, frame_height
        )
        left_hip_px = _get_landmark_pixel_coords(
            landmarks, mp_pose.PoseLandmark.LEFT_HIP, frame_width, frame_height
        )
        right_hip_px = _get_landmark_pixel_coords(
            landmarks, mp_pose.PoseLandmark.RIGHT_HIP, frame_width, frame_height
        )
        
        # Compute mid-points in pixel coordinates
        mid_shoulder_px = (
            (left_shoulder_px[0] + right_shoulder_px[0]) // 2,
            (left_shoulder_px[1] + right_shoulder_px[1]) // 2,
        )
        mid_hip_px = (
            (left_hip_px[0] + right_hip_px[0]) // 2,
            (left_hip_px[1] + right_hip_px[1]) // 2,
        )
        
        # Apply rotation to pixel coordinates for visualization
        # Use the compensation rotation (opposite of frame rotation)
        compensation_rot = _smoothing_state.get("display_rotation_deg", 0)
        if compensation_rot != 0:
            center_x, center_y = frame_width / 2, frame_height / 2
            
            # Translate to origin, rotate, translate back
            def rotate_pixel_point(px, py):
                px -= center_x
                py -= center_y
                if compensation_rot == 90:
                    px, py = -py, px
                elif compensation_rot == 180:
                    px, py = -px, -py
                elif compensation_rot == 270:
                    px, py = py, -px
                px += center_x
                py += center_y
                return (int(px), int(py))
            
            mid_shoulder_px = rotate_pixel_point(mid_shoulder_px[0], mid_shoulder_px[1])
            mid_hip_px = rotate_pixel_point(mid_hip_px[0], mid_hip_px[1])
        knee_px = _get_landmark_pixel_coords(
            landmarks, mp_pose.PoseLandmark.LEFT_KNEE, frame_width, frame_height
        )
        ankle_px = _get_landmark_pixel_coords(
            landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, frame_width, frame_height
        )
        # Use robust foot keypoint for pixel coordinates too
        if foot_norm is not None:
            foot_px = (
                int(foot_norm[0] * frame_width),
                int(foot_norm[1] * frame_height),
            )
        else:
            # Fallback to foot_index even if low confidence (for drawing only)
            try:
                foot_px = _get_landmark_pixel_coords(
                    landmarks, mp_pose.PoseLandmark.LEFT_FOOT_INDEX, frame_width, frame_height
                )
            except (IndexError, AttributeError):
                foot_px = None
        elbow_px = _get_landmark_pixel_coords(
            landmarks, mp_pose.PoseLandmark.LEFT_ELBOW, frame_width, frame_height
        )
        wrist_px = _get_landmark_pixel_coords(
            landmarks, mp_pose.PoseLandmark.LEFT_WRIST, frame_width, frame_height
        )
    except (IndexError, AttributeError):
        return {"pose_detected": False}

    knee_angle = _compute_angle_deg(left_hip_norm, knee_norm, ankle_norm)
    hip_angle = _compute_angle_deg(left_shoulder_norm, left_hip_norm, knee_norm)
    
    # Compute foot angle with robust keypoint detection
    foot_angle = None
    if foot_norm is not None:
        foot_angle_raw = _compute_angle_deg(knee_norm, ankle_norm, foot_norm)  # plantar/dorsi vs shank
        if not math.isnan(foot_angle_raw):
            # Smooth foot angle with EMA and outlier rejection
            foot_angle = _smooth_angle_with_ema(
                foot_angle_raw, _smoothing_state["foot_angle_ema"], alpha=0.2, outlier_threshold=20.0
            )
            _smoothing_state["foot_angle_ema"] = foot_angle
    
    # Apply display rotation to torso points before computing angle
    # This ensures we measure relative to the displayed video's horizontal
    rot_deg = _smoothing_state.get("display_rotation_deg", 0)
    
    # IMPORTANT: If frame was rotated for display, we need to rotate landmarks in OPPOSITE direction
    # If frame rotated 90° clockwise, rotate landmarks 270° clockwise (or -90°) to compensate
    compensation_rot = (360 - rot_deg) % 360 if rot_deg != 0 else 0
    
    # Auto-detect rotation if torso angle is suspiciously high (>70°) or too low (<5°)
    # This handles cases where video is landscape but still rotated
    if compensation_rot == 0:
        # Try computing angle without rotation first
        torso_angle_test, dx_test, dy_test = _compute_torso_angle_deg(mid_hip_norm, mid_shoulder_norm)
        
        # Check if angle is suspicious (too high or too low)
        if not math.isnan(torso_angle_test) and (torso_angle_test > 70 or torso_angle_test < 5):
            # Try all rotation options to find the most reasonable angle (10-60° range)
            best_rot = 0
            best_angle = torso_angle_test
            best_score = abs(torso_angle_test - 35)  # Prefer angles around 35° (typical aero)
            
            for test_rot in [90, 180, 270]:
                mid_hip_rot = _rotate_point_norm(mid_hip_norm, test_rot)
                mid_shoulder_rot = _rotate_point_norm(mid_shoulder_norm, test_rot)
                torso_angle_rot, _, _ = _compute_torso_angle_deg(mid_hip_rot, mid_shoulder_rot)
                
                if not math.isnan(torso_angle_rot):
                    # Score: prefer angles in reasonable range (10-60°)
                    if 10 <= torso_angle_rot <= 60:
                        score = abs(torso_angle_rot - 35)
                        if score < best_score:
                            best_rot = test_rot
                            best_angle = torso_angle_rot
                            best_score = score
                    elif abs(torso_angle_rot - best_angle) > 20:
                        # If this gives a much different angle, consider it
                        score = abs(torso_angle_rot - 35)
                        if score < best_score:
                            best_rot = test_rot
                            best_angle = torso_angle_rot
                            best_score = score
            
            if best_rot != 0:
                compensation_rot = best_rot
                _smoothing_state["display_rotation_deg"] = best_rot
                print(f"Auto-detected {best_rot}° compensation rotation: torso angle {torso_angle_test:.1f}° -> {best_angle:.1f}°")
    
    mid_hip_rotated = _rotate_point_norm(mid_hip_norm, compensation_rot)
    mid_shoulder_rotated = _rotate_point_norm(mid_shoulder_norm, compensation_rot)
    
    # Compute torso angle with debugging info (using rotated mid-points)
    torso_angle_raw, dx, dy = _compute_torso_angle_deg(mid_hip_rotated, mid_shoulder_rotated)
    
    # Sanity checks
    if abs(dx) < 1e-3:
        print(f"WARNING: Torso dx ≈ 0 ({dx:.6f}), possible rotation issue or wrong landmarks")
    if not math.isnan(torso_angle_raw) and torso_angle_raw > 80:
        print(f"WARNING: Torso angle very high ({torso_angle_raw:.1f}°), dx={dx:.3f}, dy={dy:.3f}, rot={rot_deg}°")
    
    # Store debug info including rotation
    _smoothing_state["torso_debug"] = {
        "dx": dx,
        "dy": dy,
        "raw_angle": torso_angle_raw,
        "rotation_deg": compensation_rot,
        "frame_rotation_deg": rot_deg,
    }
    
    if not math.isnan(torso_angle_raw):
        # Smooth torso angle with EMA and outlier rejection (tighter threshold for torso)
        torso_angle = _smooth_angle_with_ema(
            torso_angle_raw, _smoothing_state["torso_angle_ema"], alpha=0.2, outlier_threshold=15.0
        )
        _smoothing_state["torso_angle_ema"] = torso_angle
    else:
        torso_angle = None
    
    elbow_angle = _compute_angle_deg(left_shoulder_norm, elbow_norm, wrist_norm)  # internal angle at elbow

    if (
        math.isnan(knee_angle)
        or math.isnan(hip_angle)
        or math.isnan(elbow_angle)
    ):
        return {"pose_detected": False}
    
    # Foot and torso angles are optional (can be None if not detected)
    if foot_angle is not None and math.isnan(foot_angle):
        foot_angle = None
    if torso_angle is not None and math.isnan(torso_angle):
        torso_angle = None

    result = {
        "pose_detected": True,
        "knee_angle_deg": round(knee_angle, 2),
        "hip_angle_deg": round(hip_angle, 2),
        "elbow_angle_deg": round(elbow_angle, 2),
        "landmarks_px": {
            "shoulder": left_shoulder_px,
            "hip": left_hip_px,
            "mid_shoulder": mid_shoulder_px,
            "mid_hip": mid_hip_px,
            "knee": knee_px,
            "ankle": ankle_px,
            "elbow": elbow_px,
            "wrist": wrist_px,
        },
    }
    
    # Add optional angles (can be None if not detected)
    if foot_angle is not None:
        result["foot_angle_deg"] = round(foot_angle, 2)
    else:
        result["foot_angle_deg"] = None
    
    if torso_angle is not None:
        result["torso_angle_deg"] = round(torso_angle, 2)
        # Add debug info for torso
        debug_info = _smoothing_state.get("torso_debug", {})
        result["torso_debug"] = {
            "dx": round(debug_info.get("dx", 0), 4),
            "dy": round(debug_info.get("dy", 0), 4),
            "raw_angle": round(debug_info.get("raw_angle", 0), 2) if debug_info.get("raw_angle") is not None else None,
            "rotation_deg": debug_info.get("rotation_deg", 0),
        }
    else:
        result["torso_angle_deg"] = None
        result["torso_debug"] = None
    
    # Add foot pixel coordinates if available
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

    # Torso angle label and debugging visualization (using mid-torso points)
    if torso_angle_deg is not None:
        # Get mid-torso points for visualization
        mid_shoulder = landmarks_px.get("mid_shoulder")
        mid_hip = landmarks_px.get("mid_hip")
        
        if mid_shoulder is not None and mid_hip is not None:
            # Draw torso line (mid_hip → mid_shoulder) in distinct color for debugging
            torso_line_color = (255, 0, 255)  # Magenta in BGR for visibility
            cv2.line(annotated, mid_hip, mid_shoulder, torso_line_color, 4)
            
            # Get debug info if available
            debug_info = _smoothing_state.get("torso_debug", {})
            dx = debug_info.get("dx", 0)
            dy = debug_info.get("dy", 0)
            compensation_rot = debug_info.get("rotation_deg", 0)
            frame_rot = debug_info.get("frame_rotation_deg", 0)
            
            torso_mid_x = (mid_hip[0] + mid_shoulder[0]) // 2
            torso_mid_y = (mid_hip[1] + mid_shoulder[1]) // 2
            
            # Display angle with explicit label
            torso_text = f"Torso (to horizontal): {torso_angle_deg:.1f}°"
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
            
            # Debug text below (smaller font) - shows rotation and dx/dy
            debug_text = f"comp_rot={compensation_rot}° frame_rot={frame_rot}° dx={dx:.3f} dy={dy:.3f}"
            debug_text_size = cv2.getTextSize(debug_text, font, font_scale * 0.7, 1)[0]
            debug_text_x = torso_mid_x - debug_text_size[0] // 2
            debug_text_y = torso_mid_y + 15
            cv2.putText(annotated, debug_text, (debug_text_x, debug_text_y), font, font_scale * 0.7, text_color, 1)

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

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

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

            # Process every Nth frame
            if frame_idx % sample_every_n_frames == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    # Analyze pose and get angles/landmarks
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

                        # Draw overlay on frame
                        annotated_frame = draw_pose_overlay(
                            frame_rgb,
                            landmarks_px,
                            knee_angle,
                            hip_angle,
                            foot_angle_deg=foot_angle,
                            torso_angle_deg=torso_angle,
                            elbow_angle_deg=elbow_angle,
                        )
                        # Convert RGB back to BGR for video writer
                        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                        out.write(annotated_frame_bgr)
                    else:
                        # No pose detected, write original frame
                        out.write(frame_bgr)
                else:
                    # No pose detected, write original frame
                    out.write(frame_bgr)
            else:
                # Skip frame, write original
                out.write(frame_bgr)

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

