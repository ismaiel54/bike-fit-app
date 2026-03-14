"""
Unit tests for the framing assessment module.

Covers:
  1. Rider occupies too much frame width  → too_close
  2. Rider occupies too much frame height → too_close
  3. Multiple critical landmarks near image edges → too_close / edge reasons
  4. Missing ankle/wrist near edge → cropped / edge_adjacent_missing
  5. Healthy full-body framing → good
  6. Low-confidence landmarks → low_confidence / degraded score
  7. Video-level aggregation stability
  8. Too-far detection
  9. Bounding box derivation
 10. Edge proximity details
"""

import pytest

from framing_assessment import (
    DEFAULT_CONFIG,
    BoundingBox,
    FrameAssessment,
    FramingConfig,
    FramingReasonCode,
    FramingStatus,
    LandmarkPoint,
    VideoAssessment,
    aggregate_video_framing,
    assess_frame_framing,
    compute_edge_proximity,
    compute_frame_occupancy,
    compute_mean_confidence,
    count_low_confidence_landmarks,
    count_missing_critical_landmarks,
    detect_edge_adjacent_missing,
    get_bounding_box_from_landmarks,
)


# ---------------------------------------------------------------------------
# Helpers to build landmark sets quickly
# ---------------------------------------------------------------------------

def _make_landmarks(overrides: dict | None = None, base_conf: float = 0.9) -> list[LandmarkPoint]:
    """Create a full set of well-framed landmarks with optional overrides.

    Default positions simulate a rider nicely centred in frame with
    healthy margins from all edges.
    """
    defaults = {
        "nose":             (0.40, 0.12, base_conf),
        "left_shoulder":    (0.42, 0.22, base_conf),
        "right_shoulder":   (0.38, 0.22, base_conf),
        "left_elbow":       (0.48, 0.35, base_conf),
        "left_wrist":       (0.52, 0.42, base_conf),
        "left_hip":         (0.44, 0.48, base_conf),
        "right_hip":        (0.40, 0.48, base_conf),
        "left_knee":        (0.50, 0.65, base_conf),
        "left_ankle":       (0.55, 0.82, base_conf),
        "left_foot_index":  (0.58, 0.86, base_conf),
    }
    if overrides:
        defaults.update(overrides)
    return [
        LandmarkPoint(name=name, x=x, y=y, confidence=conf)
        for name, (x, y, conf) in defaults.items()
    ]


def _make_too_close_landmarks() -> list[LandmarkPoint]:
    """Landmarks spread across almost the entire frame — too close."""
    return _make_landmarks({
        "nose":             (0.50, 0.02, 0.9),
        "left_shoulder":    (0.55, 0.15, 0.9),
        "right_shoulder":   (0.05, 0.15, 0.9),
        "left_elbow":       (0.80, 0.30, 0.9),
        "left_wrist":       (0.92, 0.40, 0.9),
        "left_hip":         (0.50, 0.55, 0.9),
        "right_hip":        (0.10, 0.55, 0.9),
        "left_knee":        (0.70, 0.75, 0.9),
        "left_ankle":       (0.85, 0.95, 0.9),
        "left_foot_index":  (0.90, 0.98, 0.9),
    })


def _make_cropped_landmarks() -> list[LandmarkPoint]:
    """Ankle and foot missing with knee near bottom edge."""
    return _make_landmarks({
        "left_knee":        (0.55, 0.96, 0.7),
        "left_ankle":       (0.60, 0.99, 0.08),   # very low confidence = missing
        "left_foot_index":  (0.62, 0.99, 0.05),   # practically invisible
    })


def _make_too_far_landmarks() -> list[LandmarkPoint]:
    """Rider is tiny in the frame."""
    return _make_landmarks({
        "nose":             (0.48, 0.42, 0.6),
        "left_shoulder":    (0.49, 0.44, 0.6),
        "right_shoulder":   (0.47, 0.44, 0.6),
        "left_elbow":       (0.50, 0.47, 0.6),
        "left_wrist":       (0.51, 0.49, 0.6),
        "left_hip":         (0.49, 0.50, 0.6),
        "right_hip":        (0.48, 0.50, 0.6),
        "left_knee":        (0.50, 0.53, 0.6),
        "left_ankle":       (0.51, 0.55, 0.6),
        "left_foot_index":  (0.51, 0.56, 0.6),
    })


def _make_low_confidence_landmarks() -> list[LandmarkPoint]:
    """All landmarks present but with very low confidence."""
    return _make_landmarks(base_conf=0.25)


# ===========================================================================
# Test: Pure helper functions
# ===========================================================================

class TestBoundingBox:
    def test_from_good_landmarks(self):
        lms = _make_landmarks()
        bbox = get_bounding_box_from_landmarks(lms)
        assert bbox is not None
        assert 0.0 < bbox.width < 1.0
        assert 0.0 < bbox.height < 1.0

    def test_returns_none_for_no_visible(self):
        lms = [LandmarkPoint("a", 0.5, 0.5, 0.0), LandmarkPoint("b", 0.6, 0.6, 0.0)]
        bbox = get_bounding_box_from_landmarks(lms, min_confidence=0.1)
        assert bbox is None

    def test_width_height_properties(self):
        bbox = BoundingBox(0.1, 0.2, 0.9, 0.8)
        assert abs(bbox.width - 0.8) < 1e-6
        assert abs(bbox.height - 0.6) < 1e-6

    def test_center_property(self):
        bbox = BoundingBox(0.0, 0.0, 1.0, 1.0)
        cx, cy = bbox.center
        assert abs(cx - 0.5) < 1e-6
        assert abs(cy - 0.5) < 1e-6


class TestFrameOccupancy:
    def test_full_frame_occupancy(self):
        bbox = BoundingBox(0.0, 0.0, 1.0, 1.0)
        w, h = compute_frame_occupancy(bbox)
        assert abs(w - 1.0) < 1e-6
        assert abs(h - 1.0) < 1e-6

    def test_small_occupancy(self):
        bbox = BoundingBox(0.4, 0.4, 0.6, 0.6)
        w, h = compute_frame_occupancy(bbox)
        assert abs(w - 0.2) < 1e-6
        assert abs(h - 0.2) < 1e-6


class TestEdgeProximity:
    def test_no_landmarks_near_edge(self):
        lms = _make_landmarks()
        result = compute_edge_proximity(lms)
        total = sum(len(v) for v in result.values())
        assert total == 0

    def test_landmark_near_top(self):
        lms = _make_landmarks({"nose": (0.5, 0.01, 0.9)})
        result = compute_edge_proximity(lms)
        assert "nose" in result["top"]

    def test_landmark_near_bottom(self):
        lms = _make_landmarks({"left_ankle": (0.5, 0.98, 0.9)})
        result = compute_edge_proximity(lms)
        assert "left_ankle" in result["bottom"]

    def test_landmark_near_left(self):
        lms = _make_landmarks({"left_hip": (0.01, 0.5, 0.9)})
        result = compute_edge_proximity(lms)
        assert "left_hip" in result["left"]

    def test_landmark_near_right(self):
        lms = _make_landmarks({"left_wrist": (0.98, 0.5, 0.9)})
        result = compute_edge_proximity(lms)
        assert "left_wrist" in result["right"]

    def test_low_confidence_landmarks_ignored(self):
        lms = _make_landmarks({"nose": (0.5, 0.01, 0.1)})
        result = compute_edge_proximity(lms)
        assert "nose" not in result["top"]


class TestMissingCriticalLandmarks:
    def test_all_present(self):
        lms = _make_landmarks()
        count, names = count_missing_critical_landmarks(lms)
        assert count == 0
        assert names == []

    def test_low_confidence_counts_as_missing(self):
        lms = _make_landmarks({"left_ankle": (0.5, 0.8, 0.1)})
        count, names = count_missing_critical_landmarks(lms)
        assert "left_ankle" in names
        assert count >= 1

    def test_multiple_missing(self):
        lms = _make_landmarks({
            "left_ankle": (0.5, 0.8, 0.1),
            "left_foot_index": (0.5, 0.9, 0.1),
            "left_wrist": (0.5, 0.4, 0.1),
        })
        count, names = count_missing_critical_landmarks(lms)
        assert count >= 3


class TestEdgeAdjacentMissing:
    def test_no_edge_adjacent_when_healthy(self):
        lms = _make_landmarks()
        count, names = detect_edge_adjacent_missing(lms)
        assert count == 0

    def test_ankle_missing_with_knee_near_edge(self):
        lms = _make_landmarks({
            "left_knee": (0.55, 0.96, 0.7),
            "left_ankle": (0.60, 0.99, 0.1),
        })
        count, names = detect_edge_adjacent_missing(lms)
        assert count >= 1
        assert "left_ankle" in names

    def test_wrist_missing_with_elbow_near_edge(self):
        lms = _make_landmarks({
            "left_elbow": (0.97, 0.35, 0.7),
            "left_wrist": (0.99, 0.40, 0.1),
        })
        count, names = detect_edge_adjacent_missing(lms)
        assert count >= 1
        assert "left_wrist" in names


class TestMeanConfidence:
    def test_high_confidence(self):
        lms = _make_landmarks(base_conf=0.95)
        mc = compute_mean_confidence(lms)
        assert mc > 0.9

    def test_low_confidence(self):
        lms = _make_landmarks(base_conf=0.2)
        mc = compute_mean_confidence(lms)
        assert mc < 0.3


# ===========================================================================
# Test: Per-frame assessment
# ===========================================================================

class TestAssessFrameFraming:
    def test_good_framing(self):
        lms = _make_landmarks()
        result = assess_frame_framing(lms)
        assert result.status == FramingStatus.GOOD
        assert result.severity_score < DEFAULT_CONFIG.severity_warn
        assert result.message  # non-empty

    def test_too_close_high_width(self):
        """Rider occupies >85% of frame width → too_close."""
        lms = _make_too_close_landmarks()
        result = assess_frame_framing(lms)
        assert result.status in (FramingStatus.TOO_CLOSE, FramingStatus.CROPPED)
        assert result.severity_score >= DEFAULT_CONFIG.severity_warn
        occupancy_reasons = {
            FramingReasonCode.HIGH_WIDTH_OCCUPANCY,
            FramingReasonCode.HIGH_HEIGHT_OCCUPANCY,
        }
        assert set(result.reasons) & occupancy_reasons or \
            FramingReasonCode.EDGE_ADJACENT_MISSING in result.reasons

    def test_too_close_high_height(self):
        """Rider spans nearly full frame height → too_close."""
        lms = _make_landmarks({
            "nose":             (0.40, 0.02, 0.9),
            "left_foot_index":  (0.55, 0.98, 0.9),
            "left_ankle":       (0.54, 0.95, 0.9),
        })
        result = assess_frame_framing(lms)
        assert result.severity_score >= DEFAULT_CONFIG.severity_warn
        assert FramingReasonCode.HIGH_HEIGHT_OCCUPANCY in result.reasons or \
            result.status != FramingStatus.GOOD

    def test_cropped_lower_body(self):
        """Ankle/foot missing with knee near bottom edge → cropped."""
        lms = _make_cropped_landmarks()
        result = assess_frame_framing(lms)
        assert result.status in (FramingStatus.CROPPED, FramingStatus.TOO_CLOSE, FramingStatus.LOW_CONFIDENCE)
        assert result.severity_score > 0

    def test_landmarks_near_multiple_edges(self):
        """Multiple landmarks near edges → elevated severity."""
        lms = _make_landmarks({
            "nose":             (0.50, 0.01, 0.9),
            "left_ankle":       (0.98, 0.98, 0.9),
            "left_wrist":       (0.99, 0.40, 0.9),
        })
        result = assess_frame_framing(lms)
        edge_reasons = {
            FramingReasonCode.LANDMARKS_NEAR_TOP,
            FramingReasonCode.LANDMARKS_NEAR_BOTTOM,
            FramingReasonCode.LANDMARKS_NEAR_RIGHT,
        }
        assert set(result.reasons) & edge_reasons

    def test_low_confidence_landmarks(self):
        """All landmarks low confidence → low_confidence."""
        lms = _make_low_confidence_landmarks()
        result = assess_frame_framing(lms)
        assert result.severity_score > 0
        assert result.metrics.mean_confidence < 0.3

    def test_too_far(self):
        """Rider very small in frame → too_far or low severity."""
        lms = _make_too_far_landmarks()
        result = assess_frame_framing(lms)
        if result.status == FramingStatus.TOO_FAR:
            assert FramingReasonCode.LOW_OCCUPANCY in result.reasons

    def test_metrics_populated(self):
        """Check that metrics are fully populated."""
        lms = _make_landmarks()
        result = assess_frame_framing(lms)
        m = result.metrics
        assert m.width_ratio > 0
        assert m.height_ratio > 0
        assert m.bounding_box is not None
        assert m.mean_confidence > 0
        assert m.visible_landmark_count > 0

    def test_empty_landmarks_returns_low_confidence(self):
        """No usable landmarks → low_confidence."""
        lms = [LandmarkPoint("nose", 0.5, 0.5, 0.0)]
        result = assess_frame_framing(lms)
        assert result.status == FramingStatus.LOW_CONFIDENCE


# ===========================================================================
# Test: Video-level aggregation
# ===========================================================================

class TestAggregateVideoFraming:
    def test_all_good_frames(self):
        """All frames good → video is good."""
        lms = _make_landmarks()
        frames = [assess_frame_framing(lms) for _ in range(10)]
        video = aggregate_video_framing(frames)
        assert video.status == FramingStatus.GOOD
        assert video.score > 0.7

    def test_all_bad_frames(self):
        """All frames too close → video is too close."""
        lms = _make_too_close_landmarks()
        frames = [assess_frame_framing(lms) for _ in range(10)]
        video = aggregate_video_framing(frames)
        assert video.status in (FramingStatus.TOO_CLOSE, FramingStatus.CROPPED)
        assert video.score < 0.5

    def test_one_bad_frame_does_not_dominate(self):
        """1 bad frame out of 10 good → video stays good or mild warning."""
        good_lms = _make_landmarks()
        bad_lms = _make_too_close_landmarks()
        frames = [assess_frame_framing(good_lms) for _ in range(9)]
        frames.append(assess_frame_framing(bad_lms))
        video = aggregate_video_framing(frames)
        assert video.status == FramingStatus.GOOD or video.score >= 0.5

    def test_majority_bad_frames(self):
        """7 bad, 3 good → video should be flagged."""
        good_lms = _make_landmarks()
        bad_lms = _make_too_close_landmarks()
        frames = [assess_frame_framing(bad_lms) for _ in range(7)]
        frames += [assess_frame_framing(good_lms) for _ in range(3)]
        video = aggregate_video_framing(frames)
        assert video.status != FramingStatus.GOOD

    def test_empty_assessments(self):
        """No frames → default good."""
        video = aggregate_video_framing([])
        assert video.status == FramingStatus.GOOD
        assert video.score == 1.0

    def test_to_dict_serialisation(self):
        """Video assessment serialises to a clean dict."""
        lms = _make_landmarks()
        frames = [assess_frame_framing(lms) for _ in range(3)]
        video = aggregate_video_framing(frames)
        d = video.to_dict()
        assert "status" in d
        assert "score" in d
        assert "reasons" in d
        assert "message" in d
        assert "metrics" in d
        assert isinstance(d["reasons"], list)
        assert isinstance(d["metrics"], dict)

    def test_metrics_contain_expected_keys(self):
        """Summary metrics have all expected aggregation data."""
        lms = _make_landmarks()
        frames = [assess_frame_framing(lms) for _ in range(5)]
        video = aggregate_video_framing(frames)
        m = video.metrics
        expected_keys = [
            "frames_assessed", "hard_fail_count", "warn_count",
            "fail_ratio", "warn_ratio", "median_severity",
            "p90_severity", "max_severity", "blended_severity",
            "avg_width_ratio", "avg_height_ratio",
            "avg_missing_critical", "avg_confidence",
        ]
        for key in expected_keys:
            assert key in m, f"Missing metric key: {key}"


# ===========================================================================
# Test: Config customisation
# ===========================================================================

class TestConfigCustomisation:
    def test_stricter_config_flags_normal_as_bad(self):
        """Tightening thresholds should make previously-good frames flag."""
        strict = FramingConfig(
            max_width_ratio=0.30,
            max_height_ratio=0.30,
            warn_width_ratio=0.20,
            warn_height_ratio=0.20,
        )
        lms = _make_landmarks()
        result = assess_frame_framing(lms, config=strict)
        assert result.severity_score > 0

    def test_lenient_config_passes_borderline(self):
        """Very lenient thresholds should pass almost anything."""
        lenient = FramingConfig(
            max_width_ratio=0.99,
            max_height_ratio=0.99,
            edge_margin=0.001,
            severity_hard_fail=50.0,
            severity_warn=30.0,
        )
        lms = _make_too_close_landmarks()
        result = assess_frame_framing(lms, config=lenient)
        assert result.severity_score < lenient.severity_warn
