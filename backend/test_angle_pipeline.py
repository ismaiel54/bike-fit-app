"""
Tests for the hardened torso / hip angle pipeline.

Covers:
  1. Torso angle on left-side rider
  2. Torso angle on right-side rider
  3. Hip angle on left-side rider
  4. Hip angle on right-side rider
  5. Portrait-origin video normalised to landscape (no double-rotation)
  6. Side selection chooses higher-confidence side
  7. Low-confidence landmarks degrade reliability
  8. Torso angle is NOT silently forced toward expected values
  9. Consistent definitions across frames
 10. Geometry edge cases
"""

import math
from unittest.mock import MagicMock

import pytest

from angles import (
    compute_signed_segment_angle_deg,
    compute_torso_to_horizontal_deg,
)

from pose_analysis import (
    MIN_LANDMARK_CONFIDENCE,
    SIDE_LANDMARK_ENUMS,
    _compute_angle_deg,
    get_metric_reliability,
    select_analysis_side,
)


# ---------------------------------------------------------------------------
# Helpers to build mock MediaPipe landmarks
# ---------------------------------------------------------------------------

def _make_landmark(x: float, y: float, visibility: float):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.visibility = visibility
    return lm


def _make_landmarks_obj(landmark_dict: dict):
    """Build a mock MediaPipe landmarks object.

    landmark_dict maps PoseLandmark enum values to (x, y, visibility) tuples.
    Landmarks not in the dict get (0, 0, 0).
    """
    max_idx = max(landmark_dict.keys()) if landmark_dict else 0
    lm_list = []
    for i in range(max_idx + 1):
        if i in landmark_dict:
            x, y, v = landmark_dict[i]
            lm_list.append(_make_landmark(x, y, v))
        else:
            lm_list.append(_make_landmark(0.0, 0.0, 0.0))

    obj = MagicMock()
    obj.landmark = lm_list
    return obj


def _full_left_side_landmarks(base_vis: float = 0.95):
    """Landmarks for a rider visible from the LEFT side, facing right."""
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    return {
        mp_pose.PoseLandmark.NOSE: (0.45, 0.15, base_vis),
        mp_pose.PoseLandmark.LEFT_SHOULDER: (0.42, 0.25, base_vis),
        mp_pose.PoseLandmark.LEFT_HIP: (0.40, 0.50, base_vis),
        mp_pose.PoseLandmark.LEFT_KNEE: (0.55, 0.70, base_vis),
        mp_pose.PoseLandmark.LEFT_ANKLE: (0.50, 0.88, base_vis),
        mp_pose.PoseLandmark.LEFT_ELBOW: (0.50, 0.35, base_vis),
        mp_pose.PoseLandmark.LEFT_WRIST: (0.55, 0.42, base_vis),
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX: (0.52, 0.92, base_vis),
        # Right side barely visible (far side)
        mp_pose.PoseLandmark.RIGHT_SHOULDER: (0.38, 0.25, 0.2),
        mp_pose.PoseLandmark.RIGHT_HIP: (0.36, 0.50, 0.2),
        mp_pose.PoseLandmark.RIGHT_KNEE: (0.53, 0.70, 0.15),
        mp_pose.PoseLandmark.RIGHT_ANKLE: (0.48, 0.88, 0.1),
        mp_pose.PoseLandmark.RIGHT_ELBOW: (0.48, 0.35, 0.15),
        mp_pose.PoseLandmark.RIGHT_WRIST: (0.53, 0.42, 0.1),
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX: (0.50, 0.92, 0.1),
    }


def _full_right_side_landmarks(base_vis: float = 0.95):
    """Landmarks for a rider visible from the RIGHT side, facing left."""
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    return {
        mp_pose.PoseLandmark.NOSE: (0.55, 0.15, base_vis),
        mp_pose.PoseLandmark.RIGHT_SHOULDER: (0.58, 0.25, base_vis),
        mp_pose.PoseLandmark.RIGHT_HIP: (0.60, 0.50, base_vis),
        mp_pose.PoseLandmark.RIGHT_KNEE: (0.45, 0.70, base_vis),
        mp_pose.PoseLandmark.RIGHT_ANKLE: (0.50, 0.88, base_vis),
        mp_pose.PoseLandmark.RIGHT_ELBOW: (0.50, 0.35, base_vis),
        mp_pose.PoseLandmark.RIGHT_WRIST: (0.45, 0.42, base_vis),
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX: (0.48, 0.92, base_vis),
        # Left side barely visible (far side)
        mp_pose.PoseLandmark.LEFT_SHOULDER: (0.62, 0.25, 0.2),
        mp_pose.PoseLandmark.LEFT_HIP: (0.64, 0.50, 0.2),
        mp_pose.PoseLandmark.LEFT_KNEE: (0.47, 0.70, 0.15),
        mp_pose.PoseLandmark.LEFT_ANKLE: (0.52, 0.88, 0.1),
        mp_pose.PoseLandmark.LEFT_ELBOW: (0.52, 0.35, 0.15),
        mp_pose.PoseLandmark.LEFT_WRIST: (0.47, 0.42, 0.1),
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX: (0.50, 0.92, 0.1),
    }


# ===========================================================================
# Test: compute_signed_segment_angle_deg
# ===========================================================================

class TestSignedSegmentAngle:
    def test_straight_right(self):
        """Vector pointing right along x-axis → 0°."""
        assert abs(compute_signed_segment_angle_deg((0, 0.5), (1, 0.5))) < 0.01

    def test_straight_up_image_coords(self):
        """Vector pointing up in image coords (decreasing y) → +90°."""
        angle = compute_signed_segment_angle_deg((0.5, 0.8), (0.5, 0.2))
        assert abs(angle - 90.0) < 0.01

    def test_straight_down_image_coords(self):
        """Vector pointing down in image coords (increasing y) → -90°."""
        angle = compute_signed_segment_angle_deg((0.5, 0.2), (0.5, 0.8))
        assert abs(angle - (-90.0)) < 0.01

    def test_straight_left(self):
        """Vector pointing left → ±180°."""
        angle = compute_signed_segment_angle_deg((0.8, 0.5), (0.2, 0.5))
        assert abs(abs(angle) - 180.0) < 0.01

    def test_diagonal_up_right(self):
        """45° up-right in standard coords."""
        angle = compute_signed_segment_angle_deg((0.0, 1.0), (1.0, 0.0))
        assert abs(angle - 45.0) < 0.01


# ===========================================================================
# Test: compute_torso_to_horizontal_deg
# ===========================================================================

class TestTorsoToHorizontal:
    def test_rider_facing_right(self):
        """Shoulder above and to the right of hip → positive, reasonable angle."""
        angle, diag = compute_torso_to_horizontal_deg((0.4, 0.6), (0.6, 0.3))
        assert angle is not None
        assert 0 < angle < 90
        assert diag["reliable"] is True
        assert diag["shoulder_above_hip"] is True

    def test_rider_facing_left(self):
        """Shoulder above and to the left of hip → same magnitude."""
        angle_r, _ = compute_torso_to_horizontal_deg((0.4, 0.6), (0.6, 0.3))
        angle_l, diag = compute_torso_to_horizontal_deg((0.6, 0.6), (0.4, 0.3))
        assert angle_l is not None
        assert abs(angle_r - angle_l) < 0.1
        assert diag["reliable"] is True

    def test_very_aero_not_silently_changed(self):
        """Torso at ~4° should NOT be altered by any heuristic."""
        angle, diag = compute_torso_to_horizontal_deg((0.3, 0.50), (0.7, 0.47))
        assert angle is not None
        assert angle < 10.0
        assert diag["reliable"] is True

    def test_upright_rider(self):
        """Nearly vertical torso → close to 90°."""
        angle, _ = compute_torso_to_horizontal_deg((0.5, 0.8), (0.5, 0.2))
        assert angle is not None
        assert angle > 85.0

    def test_perfectly_horizontal(self):
        """Perfectly flat torso → 0°."""
        angle, _ = compute_torso_to_horizontal_deg((0.3, 0.5), (0.7, 0.5))
        assert angle is not None
        assert angle < 0.1

    def test_shoulder_below_hip_unreliable(self):
        """If shoulder is below hip in image, mark unreliable."""
        angle, diag = compute_torso_to_horizontal_deg((0.5, 0.3), (0.5, 0.8))
        assert angle is not None
        assert diag["reliable"] is False
        assert diag["shoulder_above_hip"] is False

    def test_zero_length_returns_none(self):
        angle, diag = compute_torso_to_horizontal_deg((0.5, 0.5), (0.5, 0.5))
        assert angle is None
        assert diag["valid"] is False

    def test_extreme_angle_preserved(self):
        """A 3° torso angle should be returned as-is, not forced to 35°."""
        angle, _ = compute_torso_to_horizontal_deg((0.2, 0.51), (0.8, 0.48))
        assert angle is not None
        assert angle < 5.0

    def test_portrait_rotation_compensated(self):
        """A portrait video rotated 90° CW should give the same torso angle
        as the original un-rotated orientation.

        Original: hip=(0.50,0.50) shoulder=(0.65,0.47) → ~11° from horizontal.
        After 90° CW: (x,y)→(1-y,x) → hip=(0.50,0.50) shoulder=(0.53,0.65).
        Without compensation the rotated vector gives ~79°. With frame_rotation_deg=90
        it should recover ~11°.
        """
        # Original (no rotation)
        angle_orig, _ = compute_torso_to_horizontal_deg(
            (0.50, 0.50), (0.65, 0.47), frame_rotation_deg=0)

        # Rotated landmarks + compensation
        angle_comp, _ = compute_torso_to_horizontal_deg(
            (0.50, 0.50), (0.53, 0.65), frame_rotation_deg=90)

        # Rotated landmarks WITHOUT compensation → wrong
        angle_wrong, _ = compute_torso_to_horizontal_deg(
            (0.50, 0.50), (0.53, 0.65), frame_rotation_deg=0)

        assert angle_orig is not None and angle_comp is not None
        assert abs(angle_orig - angle_comp) < 1.0, (
            f"Compensated ({angle_comp:.1f}°) should match original ({angle_orig:.1f}°)")
        assert angle_wrong > 70, (
            f"Without compensation should be ~79°, got {angle_wrong:.1f}°")


# ===========================================================================
# Test: hip angle (included joint angle)
# ===========================================================================

class TestHipAngle:
    def test_right_angle_hip(self):
        """Shoulder straight up, knee straight down → 90° included angle."""
        shoulder = (0.5, 0.0)
        hip = (0.5, 0.5)
        knee = (1.0, 0.5)
        angle = _compute_angle_deg(shoulder, hip, knee)
        assert abs(angle - 90.0) < 0.1

    def test_acute_hip(self):
        """Very closed hip → small included angle.

        When shoulder and knee are on the SAME side of the hip (both above),
        the included angle is small.
        """
        shoulder = (0.6, 0.2)
        hip = (0.5, 0.5)
        knee = (0.65, 0.25)  # knee very close to shoulder direction
        angle = _compute_angle_deg(shoulder, hip, knee)
        assert angle < 30

    def test_open_hip(self):
        """Wide open hip → large angle."""
        shoulder = (0.2, 0.2)
        hip = (0.5, 0.5)
        knee = (0.8, 0.8)
        angle = _compute_angle_deg(shoulder, hip, knee)
        assert angle > 150

    def test_rotation_invariant(self):
        """Rotating all three points by the same amount doesn't change the angle."""
        s, h, k = (0.42, 0.25), (0.40, 0.50), (0.55, 0.70)
        angle_orig = _compute_angle_deg(s, h, k)

        import numpy as np
        theta = math.radians(45)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        def rot(p):
            x, y = p[0] - 0.5, p[1] - 0.5
            return (cos_t * x - sin_t * y + 0.5, sin_t * x + cos_t * y + 0.5)

        angle_rot = _compute_angle_deg(rot(s), rot(h), rot(k))
        assert abs(angle_orig - angle_rot) < 0.01

    def test_left_side_typical_road(self):
        """Typical road bike position from left side → plausible included angle."""
        shoulder = (0.42, 0.25)
        hip = (0.40, 0.50)
        knee = (0.55, 0.70)
        angle = _compute_angle_deg(shoulder, hip, knee)
        assert 60 < angle < 160  # reasonable range for a road bike hip angle

    def test_right_side_gives_same_angle(self):
        """Mirror image landmarks should give same hip angle."""
        s_l, h_l, k_l = (0.42, 0.25), (0.40, 0.50), (0.55, 0.70)
        s_r = (1.0 - s_l[0], s_l[1])
        h_r = (1.0 - h_l[0], h_l[1])
        k_r = (1.0 - k_l[0], k_l[1])
        assert abs(_compute_angle_deg(s_l, h_l, k_l) - _compute_angle_deg(s_r, h_r, k_r)) < 0.01


# ===========================================================================
# Test: select_analysis_side
# ===========================================================================

class TestSelectAnalysisSide:
    def test_left_side_chosen_when_left_visible(self):
        lm_dict = _full_left_side_landmarks()
        landmarks = _make_landmarks_obj(lm_dict)
        side, scores = select_analysis_side(landmarks)
        assert side == "left"
        assert scores["left"] > scores["right"]

    def test_right_side_chosen_when_right_visible(self):
        lm_dict = _full_right_side_landmarks()
        landmarks = _make_landmarks_obj(lm_dict)
        side, scores = select_analysis_side(landmarks)
        assert side == "right"
        assert scores["right"] > scores["left"]

    def test_equal_confidence_defaults_to_left(self):
        """When both sides have identical confidence, prefer left."""
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        lm_dict = {}
        for side_enums in SIDE_LANDMARK_ENUMS.values():
            for enum_val in side_enums.values():
                lm_dict[enum_val] = (0.5, 0.5, 0.8)
        landmarks = _make_landmarks_obj(lm_dict)
        side, _ = select_analysis_side(landmarks)
        assert side == "left"

    def test_weights_critical_joints_higher(self):
        """Shoulder/hip/knee get 2x weight — high confidence on those
        should outweigh low confidence on non-critical joints."""
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        lm_dict = {}
        for name, enum_val in SIDE_LANDMARK_ENUMS["left"].items():
            if name in ("shoulder", "hip", "knee"):
                lm_dict[enum_val] = (0.5, 0.5, 0.95)
            else:
                lm_dict[enum_val] = (0.5, 0.5, 0.1)
        for name, enum_val in SIDE_LANDMARK_ENUMS["right"].items():
            lm_dict[enum_val] = (0.5, 0.5, 0.5)
        landmarks = _make_landmarks_obj(lm_dict)
        side, scores = select_analysis_side(landmarks)
        assert side == "left"


# ===========================================================================
# Test: get_metric_reliability
# ===========================================================================

class TestMetricReliability:
    def test_all_reliable(self):
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        lm_dict = _full_left_side_landmarks(base_vis=0.9)
        landmarks = _make_landmarks_obj(lm_dict)
        ok, conf, weak = get_metric_reliability(
            landmarks,
            [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP],
        )
        assert ok is True
        assert conf > 0.8
        assert weak == []

    def test_low_confidence_flagged(self):
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        lm_dict = _full_left_side_landmarks(base_vis=0.9)
        lm_dict[mp_pose.PoseLandmark.LEFT_SHOULDER] = (0.42, 0.25, 0.1)
        landmarks = _make_landmarks_obj(lm_dict)
        ok, conf, weak = get_metric_reliability(
            landmarks,
            [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP],
        )
        assert ok is False
        assert len(weak) == 1

    def test_all_low_confidence(self):
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        lm_dict = _full_left_side_landmarks(base_vis=0.1)
        landmarks = _make_landmarks_obj(lm_dict)
        ok, conf, weak = get_metric_reliability(
            landmarks,
            [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP,
             mp_pose.PoseLandmark.LEFT_KNEE],
        )
        assert ok is False
        assert len(weak) == 3
        assert conf < MIN_LANDMARK_CONFIDENCE


# ===========================================================================
# Test: no double-rotation corruption
# ===========================================================================

class TestNoDoubleRotation:
    def test_function_uses_explicit_param_not_global(self):
        """compute_torso_to_horizontal_deg uses its frame_rotation_deg
        parameter, NOT _smoothing_state.  The global should not affect
        the result when calling the function directly."""
        from pose_analysis import _smoothing_state

        hip = (0.4, 0.6)
        shoulder = (0.5, 0.3)

        _smoothing_state["display_rotation_deg"] = 999  # garbage value
        angle, _ = compute_torso_to_horizontal_deg(hip, shoulder, frame_rotation_deg=0)
        _smoothing_state["display_rotation_deg"] = 0  # reset

        angle_clean, _ = compute_torso_to_horizontal_deg(hip, shoulder, frame_rotation_deg=0)
        assert angle is not None
        assert abs(angle - angle_clean) < 0.01

    def test_no_rotation_for_landscape_video(self):
        """With frame_rotation_deg=0, the function must not rotate anything."""
        hip = (0.4, 0.6)
        shoulder = (0.5, 0.3)
        angle, diag = compute_torso_to_horizontal_deg(hip, shoulder, frame_rotation_deg=0)
        assert angle is not None
        assert diag.get("frame_rotation_deg") == 0


# ===========================================================================
# Test: torso angle not silently forced
# ===========================================================================

class TestNoSilentForcing:
    def test_extreme_low_angle_preserved(self):
        """A 2° torso angle must come out as ~2°, not forced to 35°."""
        hip = (0.2, 0.50)
        shoulder = (0.8, 0.49)
        angle, diag = compute_torso_to_horizontal_deg(hip, shoulder)
        assert angle is not None
        assert angle < 5.0

    def test_extreme_high_angle_preserved(self):
        """An 85° torso angle must come out as ~85°, not clamped to 60°."""
        hip = (0.50, 0.90)
        shoulder = (0.51, 0.10)
        angle, _ = compute_torso_to_horizontal_deg(hip, shoulder)
        assert angle is not None
        assert angle > 80.0

    def test_values_are_deterministic(self):
        """Same inputs always produce the same output — no randomness."""
        hip, shoulder = (0.3, 0.55), (0.6, 0.30)
        results = [compute_torso_to_horizontal_deg(hip, shoulder) for _ in range(100)]
        angles = [r[0] for r in results]
        assert all(a == angles[0] for a in angles)


# ===========================================================================
# Test: consistent output definitions
# ===========================================================================

class TestConsistentDefinitions:
    def test_torso_0_means_horizontal(self):
        angle, _ = compute_torso_to_horizontal_deg((0.0, 0.5), (1.0, 0.5))
        assert angle is not None
        assert angle < 0.01

    def test_torso_90_means_vertical(self):
        angle, _ = compute_torso_to_horizontal_deg((0.5, 1.0), (0.5, 0.0))
        assert angle is not None
        assert abs(angle - 90.0) < 0.01

    def test_hip_180_means_straight_line(self):
        """Three collinear points → 180° included angle."""
        angle = _compute_angle_deg((0.0, 0.5), (0.5, 0.5), (1.0, 0.5))
        assert abs(angle - 180.0) < 0.01

    def test_hip_0_means_folded(self):
        """Shoulder and knee at the same point relative to hip → 0°."""
        angle = _compute_angle_deg((0.5, 0.0), (0.5, 0.5), (0.5, 0.0))
        assert angle < 0.01
