"""
Framing quality assessment for bike fit video analysis.

Evaluates whether the rider is properly framed in the camera view by
analyzing pose landmark positions, occupancy ratios, edge proximity,
landmark completeness, and confidence. Produces a structured assessment
that the analysis pipeline uses to warn users or gate recommendations.

Architecture:
    - FramingConfig: all thresholds in one place
    - Pure helper functions: bounding box, occupancy, edge proximity, etc.
    - assess_frame_framing(): per-frame assessment
    - aggregate_video_framing(): video-level aggregation
    - FramingStatus / FrameAssessment / VideoAssessment: typed result objects
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status and reason-code enums
# ---------------------------------------------------------------------------

class FramingStatus(str, Enum):
    """Overall framing quality classification."""
    GOOD = "good"
    TOO_CLOSE = "too_close"
    CROPPED = "cropped"
    TOO_FAR = "too_far"
    LOW_CONFIDENCE = "low_confidence"


class FramingReasonCode(str, Enum):
    """Machine-readable reason codes for framing issues."""
    HIGH_WIDTH_OCCUPANCY = "high_width_occupancy"
    HIGH_HEIGHT_OCCUPANCY = "high_height_occupancy"
    LANDMARKS_NEAR_TOP = "landmarks_near_top"
    LANDMARKS_NEAR_BOTTOM = "landmarks_near_bottom"
    LANDMARKS_NEAR_LEFT = "landmarks_near_left"
    LANDMARKS_NEAR_RIGHT = "landmarks_near_right"
    MISSING_CRITICAL_LANDMARKS = "missing_critical_landmarks"
    LOW_LANDMARK_CONFIDENCE = "low_landmark_confidence"
    EDGE_ADJACENT_MISSING = "edge_adjacent_missing"
    LOW_OCCUPANCY = "low_occupancy"
    # Future: PERSPECTIVE_DISTORTION = "perspective_distortion"


# ---------------------------------------------------------------------------
# Configuration — single source of truth for all thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FramingConfig:
    """All framing assessment thresholds. Adjust here to tune sensitivity."""

    # --- Subject occupancy ---
    max_width_ratio: float = 0.85
    max_height_ratio: float = 0.90
    warn_width_ratio: float = 0.75
    warn_height_ratio: float = 0.80
    min_width_ratio: float = 0.15
    min_height_ratio: float = 0.20

    # --- Edge proximity (fraction of frame dimension) ---
    edge_margin: float = 0.03
    edge_warn_margin: float = 0.06

    # --- Landmark confidence ---
    min_landmark_confidence: float = 0.5
    low_confidence_threshold: float = 0.3

    # --- Required landmarks for bike fit ---
    # These must be visible for a reliable analysis
    min_visible_critical_landmarks: int = 6  # out of 8 critical

    # --- Scoring weights ---
    weight_width_occupancy: float = 3.0
    weight_height_occupancy: float = 3.0
    weight_edge_proximity: float = 2.0
    weight_missing_landmarks: float = 2.5
    weight_low_confidence: float = 1.5
    weight_edge_adjacent_missing: float = 3.0

    # --- Video-level aggregation ---
    video_fail_frame_ratio: float = 0.50
    video_warn_frame_ratio: float = 0.30
    severity_hard_fail: float = 6.0
    severity_warn: float = 3.0


DEFAULT_CONFIG = FramingConfig()

# Critical landmarks for bike fit analysis (left side + head)
CRITICAL_LANDMARK_NAMES = [
    "nose",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot_index",
]


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in normalized coordinates [0, 1]."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


@dataclass
class FrameMetrics:
    """Raw metrics computed for a single frame."""
    width_ratio: float = 0.0
    height_ratio: float = 0.0
    edge_proximity_count: int = 0
    edge_proximity_details: Dict[str, List[str]] = field(default_factory=dict)
    missing_critical_count: int = 0
    missing_critical_names: List[str] = field(default_factory=list)
    low_confidence_count: int = 0
    low_confidence_names: List[str] = field(default_factory=list)
    edge_adjacent_missing_count: int = 0
    edge_adjacent_missing_names: List[str] = field(default_factory=list)
    mean_confidence: float = 0.0
    bounding_box: Optional[BoundingBox] = None
    visible_landmark_count: int = 0


@dataclass
class FrameAssessment:
    """Assessment result for a single frame."""
    status: FramingStatus
    severity_score: float
    reasons: List[FramingReasonCode]
    message: str
    metrics: FrameMetrics


@dataclass
class VideoAssessment:
    """Aggregated assessment result for an entire video."""
    status: FramingStatus
    score: float  # 0.0 (worst) to 1.0 (best) readiness confidence
    reasons: List[FramingReasonCode]
    message: str
    metrics: Dict[str, object]
    frame_assessments: List[FrameAssessment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Serialize to a JSON-compatible dict for the API response."""
        return {
            "status": self.status.value,
            "score": round(self.score, 3),
            "reasons": [r.value for r in self.reasons],
            "message": self.message,
            "metrics": self.metrics,
        }


# ---------------------------------------------------------------------------
# Landmark data adapter
# ---------------------------------------------------------------------------

@dataclass
class LandmarkPoint:
    """A single pose landmark with normalized coords and confidence."""
    name: str
    x: float  # normalized 0–1
    y: float  # normalized 0–1
    confidence: float  # visibility/confidence 0–1


def extract_landmarks_from_mediapipe(mp_landmarks, mp_pose_module) -> List[LandmarkPoint]:
    """Convert MediaPipe pose landmarks to our LandmarkPoint list.

    Args:
        mp_landmarks: mediapipe pose_landmarks result object
        mp_pose_module: mp.solutions.pose module reference

    Returns:
        List of LandmarkPoint for the landmarks we care about.
    """
    mapping = {
        "nose": mp_pose_module.PoseLandmark.NOSE,
        "left_shoulder": mp_pose_module.PoseLandmark.LEFT_SHOULDER,
        "right_shoulder": mp_pose_module.PoseLandmark.RIGHT_SHOULDER,
        "left_elbow": mp_pose_module.PoseLandmark.LEFT_ELBOW,
        "left_wrist": mp_pose_module.PoseLandmark.LEFT_WRIST,
        "left_hip": mp_pose_module.PoseLandmark.LEFT_HIP,
        "right_hip": mp_pose_module.PoseLandmark.RIGHT_HIP,
        "left_knee": mp_pose_module.PoseLandmark.LEFT_KNEE,
        "left_ankle": mp_pose_module.PoseLandmark.LEFT_ANKLE,
        "left_foot_index": mp_pose_module.PoseLandmark.LEFT_FOOT_INDEX,
    }
    points: List[LandmarkPoint] = []
    for name, enum_val in mapping.items():
        try:
            lm = mp_landmarks.landmark[enum_val]
            points.append(LandmarkPoint(
                name=name,
                x=lm.x,
                y=lm.y,
                confidence=lm.visibility,
            ))
        except (IndexError, AttributeError):
            points.append(LandmarkPoint(name=name, x=0.0, y=0.0, confidence=0.0))
    return points


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def get_bounding_box_from_landmarks(
    landmarks: List[LandmarkPoint],
    min_confidence: float = 0.1,
) -> Optional[BoundingBox]:
    """Derive an axis-aligned bounding box from visible landmarks.

    Only landmarks above *min_confidence* contribute. Returns None when
    fewer than 2 qualifying landmarks exist.
    """
    visible = [lm for lm in landmarks if lm.confidence >= min_confidence]
    if len(visible) < 2:
        return None

    xs = [lm.x for lm in visible]
    ys = [lm.y for lm in visible]
    return BoundingBox(
        x_min=min(xs),
        y_min=min(ys),
        x_max=max(xs),
        y_max=max(ys),
    )


def compute_frame_occupancy(
    bbox: BoundingBox,
) -> Tuple[float, float]:
    """Return (width_ratio, height_ratio) of the rider bounding box
    relative to the full frame (which is [0,1] x [0,1] in normalised space)."""
    return (bbox.width, bbox.height)


def compute_edge_proximity(
    landmarks: List[LandmarkPoint],
    config: FramingConfig = DEFAULT_CONFIG,
) -> Dict[str, List[str]]:
    """Identify landmarks that are dangerously close to frame edges.

    Returns a dict mapping edge names ("top", "bottom", "left", "right")
    to lists of landmark names near that edge.
    """
    proximity: Dict[str, List[str]] = {
        "top": [], "bottom": [], "left": [], "right": [],
    }
    margin = config.edge_margin
    for lm in landmarks:
        if lm.confidence < config.low_confidence_threshold:
            continue  # skip very-low-confidence landmarks
        if lm.y < margin:
            proximity["top"].append(lm.name)
        if lm.y > (1.0 - margin):
            proximity["bottom"].append(lm.name)
        if lm.x < margin:
            proximity["left"].append(lm.name)
        if lm.x > (1.0 - margin):
            proximity["right"].append(lm.name)
    return proximity


def count_missing_critical_landmarks(
    landmarks: List[LandmarkPoint],
    config: FramingConfig = DEFAULT_CONFIG,
) -> Tuple[int, List[str]]:
    """Count how many critical landmarks are missing or below confidence threshold.

    Returns (count, list_of_missing_names).
    """
    by_name = {lm.name: lm for lm in landmarks}
    missing: List[str] = []
    for name in CRITICAL_LANDMARK_NAMES:
        lm = by_name.get(name)
        if lm is None or lm.confidence < config.min_landmark_confidence:
            missing.append(name)
    return len(missing), missing


def count_low_confidence_landmarks(
    landmarks: List[LandmarkPoint],
    config: FramingConfig = DEFAULT_CONFIG,
) -> Tuple[int, List[str]]:
    """Count landmarks with confidence between low and minimum thresholds."""
    low: List[str] = []
    for lm in landmarks:
        if lm.name in CRITICAL_LANDMARK_NAMES:
            if config.low_confidence_threshold <= lm.confidence < config.min_landmark_confidence:
                low.append(lm.name)
    return len(low), low


def detect_edge_adjacent_missing(
    landmarks: List[LandmarkPoint],
    config: FramingConfig = DEFAULT_CONFIG,
) -> Tuple[int, List[str]]:
    """Detect landmarks that are likely missing *because* they are cropped
    by the frame edge.

    Heuristic: a landmark is "edge-adjacent missing" if it is below
    confidence threshold AND a neighbouring landmark on the same limb
    chain is near an edge.
    """
    by_name = {lm.name: lm for lm in landmarks}

    limb_chains = [
        ("left_shoulder", "left_elbow", "left_wrist"),
        ("left_hip", "left_knee", "left_ankle", "left_foot_index"),
        ("nose", "left_shoulder"),
    ]

    edge_margin = config.edge_warn_margin
    flagged: List[str] = []

    def _is_near_edge(lm: LandmarkPoint) -> bool:
        return (
            lm.x < edge_margin
            or lm.x > (1.0 - edge_margin)
            or lm.y < edge_margin
            or lm.y > (1.0 - edge_margin)
        )

    for chain in limb_chains:
        for i, name in enumerate(chain):
            lm = by_name.get(name)
            if lm is None or lm.confidence < config.min_landmark_confidence:
                # This landmark is missing — check neighbours
                neighbours = []
                if i > 0:
                    neighbours.append(by_name.get(chain[i - 1]))
                if i < len(chain) - 1:
                    neighbours.append(by_name.get(chain[i + 1]))

                for nb in neighbours:
                    if nb is not None and nb.confidence >= config.low_confidence_threshold and _is_near_edge(nb):
                        if name not in flagged:
                            flagged.append(name)
                        break

    return len(flagged), flagged


def compute_mean_confidence(landmarks: List[LandmarkPoint]) -> float:
    """Mean confidence across critical landmarks."""
    by_name = {lm.name: lm for lm in landmarks}
    confs = [
        by_name[n].confidence
        for n in CRITICAL_LANDMARK_NAMES
        if n in by_name
    ]
    return sum(confs) / len(confs) if confs else 0.0


# ---------------------------------------------------------------------------
# Per-frame assessment
# ---------------------------------------------------------------------------

def assess_frame_framing(
    landmarks: List[LandmarkPoint],
    config: FramingConfig = DEFAULT_CONFIG,
) -> FrameAssessment:
    """Evaluate framing quality for a single frame.

    Combines multiple signals into a severity score and classification.
    Higher severity_score = worse framing.
    """
    reasons: List[FramingReasonCode] = []
    severity = 0.0

    # 1. Bounding box & occupancy
    bbox = get_bounding_box_from_landmarks(landmarks, min_confidence=config.low_confidence_threshold)
    if bbox is None:
        return FrameAssessment(
            status=FramingStatus.LOW_CONFIDENCE,
            severity_score=10.0,
            reasons=[FramingReasonCode.MISSING_CRITICAL_LANDMARKS],
            message="Cannot assess framing — too few visible landmarks.",
            metrics=FrameMetrics(),
        )

    width_ratio, height_ratio = compute_frame_occupancy(bbox)

    if width_ratio >= config.max_width_ratio:
        severity += config.weight_width_occupancy
        reasons.append(FramingReasonCode.HIGH_WIDTH_OCCUPANCY)
    elif width_ratio >= config.warn_width_ratio:
        severity += config.weight_width_occupancy * 0.5

    if height_ratio >= config.max_height_ratio:
        severity += config.weight_height_occupancy
        reasons.append(FramingReasonCode.HIGH_HEIGHT_OCCUPANCY)
    elif height_ratio >= config.warn_height_ratio:
        severity += config.weight_height_occupancy * 0.5

    # Too far check
    if width_ratio < config.min_width_ratio and height_ratio < config.min_height_ratio:
        reasons.append(FramingReasonCode.LOW_OCCUPANCY)
        severity += 2.0

    # 2. Edge proximity
    edge_details = compute_edge_proximity(landmarks, config)
    total_edge = sum(len(v) for v in edge_details.values())
    if total_edge >= 3:
        severity += config.weight_edge_proximity
    elif total_edge >= 1:
        severity += config.weight_edge_proximity * 0.4

    for edge_name, lm_names in edge_details.items():
        if lm_names:
            code = {
                "top": FramingReasonCode.LANDMARKS_NEAR_TOP,
                "bottom": FramingReasonCode.LANDMARKS_NEAR_BOTTOM,
                "left": FramingReasonCode.LANDMARKS_NEAR_LEFT,
                "right": FramingReasonCode.LANDMARKS_NEAR_RIGHT,
            }.get(edge_name)
            if code and code not in reasons:
                reasons.append(code)

    # 3. Missing critical landmarks
    missing_count, missing_names = count_missing_critical_landmarks(landmarks, config)
    if missing_count >= 3:
        severity += config.weight_missing_landmarks
        if FramingReasonCode.MISSING_CRITICAL_LANDMARKS not in reasons:
            reasons.append(FramingReasonCode.MISSING_CRITICAL_LANDMARKS)
    elif missing_count >= 1:
        severity += config.weight_missing_landmarks * 0.4
        if FramingReasonCode.MISSING_CRITICAL_LANDMARKS not in reasons:
            reasons.append(FramingReasonCode.MISSING_CRITICAL_LANDMARKS)

    # 4. Low confidence
    low_conf_count, low_conf_names = count_low_confidence_landmarks(landmarks, config)
    if low_conf_count >= 3:
        severity += config.weight_low_confidence
        reasons.append(FramingReasonCode.LOW_LANDMARK_CONFIDENCE)
    elif low_conf_count >= 1:
        severity += config.weight_low_confidence * 0.3

    # 5. Edge-adjacent missing (strong cropping signal)
    ea_count, ea_names = detect_edge_adjacent_missing(landmarks, config)
    if ea_count >= 1:
        severity += config.weight_edge_adjacent_missing * min(ea_count, 3)
        if FramingReasonCode.EDGE_ADJACENT_MISSING not in reasons:
            reasons.append(FramingReasonCode.EDGE_ADJACENT_MISSING)

    mean_conf = compute_mean_confidence(landmarks)

    # --- Build metrics ---
    metrics = FrameMetrics(
        width_ratio=round(width_ratio, 4),
        height_ratio=round(height_ratio, 4),
        edge_proximity_count=total_edge,
        edge_proximity_details={k: v for k, v in edge_details.items() if v},
        missing_critical_count=missing_count,
        missing_critical_names=missing_names,
        low_confidence_count=low_conf_count,
        low_confidence_names=low_conf_names,
        edge_adjacent_missing_count=ea_count,
        edge_adjacent_missing_names=ea_names,
        mean_confidence=round(mean_conf, 4),
        bounding_box=bbox,
        visible_landmark_count=len(landmarks) - missing_count,
    )

    # --- Classify ---
    status, message = _classify_frame(severity, reasons, metrics, config)

    logger.debug(
        "Frame assessment: status=%s severity=%.2f reasons=%s "
        "w_ratio=%.3f h_ratio=%.3f edge=%d missing=%d ea=%d",
        status.value, severity, [r.value for r in reasons],
        width_ratio, height_ratio, total_edge, missing_count, ea_count,
    )

    return FrameAssessment(
        status=status,
        severity_score=round(severity, 3),
        reasons=reasons,
        message=message,
        metrics=metrics,
    )


def _classify_frame(
    severity: float,
    reasons: List[FramingReasonCode],
    metrics: FrameMetrics,
    config: FramingConfig,
) -> Tuple[FramingStatus, str]:
    """Determine status and user-facing message from severity + reasons."""
    occupancy_reasons = {
        FramingReasonCode.HIGH_WIDTH_OCCUPANCY,
        FramingReasonCode.HIGH_HEIGHT_OCCUPANCY,
    }
    edge_reasons = {
        FramingReasonCode.LANDMARKS_NEAR_TOP,
        FramingReasonCode.LANDMARKS_NEAR_BOTTOM,
        FramingReasonCode.LANDMARKS_NEAR_LEFT,
        FramingReasonCode.LANDMARKS_NEAR_RIGHT,
    }

    reason_set = set(reasons)

    if FramingReasonCode.LOW_OCCUPANCY in reason_set:
        return (
            FramingStatus.TOO_FAR,
            "The rider appears very small in the frame. "
            "Move the camera closer so the rider fills more of the view.",
        )

    if severity >= config.severity_hard_fail:
        if FramingReasonCode.EDGE_ADJACENT_MISSING in reason_set:
            body_part = _describe_cropped_region(metrics.edge_adjacent_missing_names)
            return (
                FramingStatus.CROPPED,
                f"Your {body_part} appears to be cut off by the frame edge. "
                "Move the camera farther back and keep your full riding position visible.",
            )

        if reason_set & occupancy_reasons:
            return (
                FramingStatus.TOO_CLOSE,
                "Camera is too close for accurate bike fit analysis. "
                "Move the camera farther back so your whole body and bike "
                "fit comfortably inside the frame.",
            )

        if reason_set & edge_reasons and metrics.missing_critical_count >= 2:
            return (
                FramingStatus.CROPPED,
                "Parts of your body are too close to the frame edges and may be partially "
                "cut off. Move the camera farther back for a complete view.",
            )

        return (
            FramingStatus.LOW_CONFIDENCE,
            "Pose detection confidence is low. Ensure good lighting, a clear "
            "side-on view, and that the full rider is visible.",
        )

    if severity >= config.severity_warn:
        if reason_set & occupancy_reasons:
            return (
                FramingStatus.TOO_CLOSE,
                "The rider may be a bit close to the camera. Results are usable "
                "but accuracy improves with more space around the rider.",
            )
        return (
            FramingStatus.LOW_CONFIDENCE,
            "Some pose landmarks have low confidence. Results may be less accurate.",
        )

    return (
        FramingStatus.GOOD,
        "Framing looks good for bike fit analysis.",
    )


def _describe_cropped_region(missing_names: List[str]) -> str:
    """Produce a human-friendly description of what's cropped."""
    lower = {"left_ankle", "left_foot_index", "left_knee"}
    upper = {"nose", "left_shoulder"}
    arm = {"left_elbow", "left_wrist"}

    names = set(missing_names)
    if names & lower:
        return "lower body"
    if names & upper:
        return "head or upper body"
    if names & arm:
        return "arm"
    return "body"


# ---------------------------------------------------------------------------
# Video-level aggregation
# ---------------------------------------------------------------------------

def aggregate_video_framing(
    frame_assessments: List[FrameAssessment],
    config: FramingConfig = DEFAULT_CONFIG,
) -> VideoAssessment:
    """Aggregate per-frame assessments into a single video-level result.

    Aggregation strategy (documented for maintainability):
      1. Compute the proportion of frames classified as problematic.
      2. Compute severity statistics: median, 90th percentile, max.
      3. Use a blend of "proportion of bad frames" and "worst-case severity"
         to avoid both over-sensitivity to one bad frame and under-sensitivity
         to widespread mild issues.
      4. Collect all reason codes that appeared in >25% of frames.
    """
    if not frame_assessments:
        return VideoAssessment(
            status=FramingStatus.GOOD,
            score=1.0,
            reasons=[],
            message="No frames to assess.",
            metrics={},
        )

    n = len(frame_assessments)
    severities = sorted([fa.severity_score for fa in frame_assessments])

    # --- Proportion of bad frames ---
    hard_fail_count = sum(
        1 for fa in frame_assessments
        if fa.severity_score >= config.severity_hard_fail
    )
    warn_count = sum(
        1 for fa in frame_assessments
        if fa.severity_score >= config.severity_warn
    )
    fail_ratio = hard_fail_count / n
    warn_ratio = warn_count / n

    # --- Severity statistics ---
    median_severity = severities[n // 2]
    p90_severity = severities[int(n * 0.9)]
    max_severity = severities[-1]

    # Blended severity: 40% median + 40% p90 + 20% max
    # This prevents one outlier frame from dominating while still
    # catching consistently bad framing.
    blended_severity = 0.4 * median_severity + 0.4 * p90_severity + 0.2 * max_severity

    # --- Collect prevalent reason codes ---
    reason_counts: Dict[FramingReasonCode, int] = {}
    for fa in frame_assessments:
        for r in fa.reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1
    prevalent_reasons = [
        r for r, count in reason_counts.items()
        if count / n >= 0.25
    ]

    # --- Classify video ---
    if fail_ratio >= config.video_fail_frame_ratio or blended_severity >= config.severity_hard_fail:
        status, message = _classify_video_bad(prevalent_reasons, frame_assessments, config)
        score = max(0.0, 1.0 - blended_severity / 12.0)
    elif warn_ratio >= config.video_warn_frame_ratio or blended_severity >= config.severity_warn:
        status, message = _classify_video_warn(prevalent_reasons)
        score = max(0.1, 1.0 - blended_severity / 10.0)
    else:
        status = FramingStatus.GOOD
        message = "Framing looks good for bike fit analysis."
        score = min(1.0, 1.0 - blended_severity / 10.0)

    # --- Collect summary metrics ---
    avg_width = sum(fa.metrics.width_ratio for fa in frame_assessments) / n
    avg_height = sum(fa.metrics.height_ratio for fa in frame_assessments) / n
    avg_missing = sum(fa.metrics.missing_critical_count for fa in frame_assessments) / n
    avg_confidence = sum(fa.metrics.mean_confidence for fa in frame_assessments) / n

    summary_metrics = {
        "frames_assessed": n,
        "hard_fail_count": hard_fail_count,
        "warn_count": warn_count,
        "fail_ratio": round(fail_ratio, 3),
        "warn_ratio": round(warn_ratio, 3),
        "median_severity": round(median_severity, 3),
        "p90_severity": round(p90_severity, 3),
        "max_severity": round(max_severity, 3),
        "blended_severity": round(blended_severity, 3),
        "avg_width_ratio": round(avg_width, 4),
        "avg_height_ratio": round(avg_height, 4),
        "avg_missing_critical": round(avg_missing, 2),
        "avg_confidence": round(avg_confidence, 4),
    }

    return VideoAssessment(
        status=status,
        score=round(max(0.0, min(1.0, score)), 3),
        reasons=prevalent_reasons,
        message=message,
        metrics=summary_metrics,
        frame_assessments=frame_assessments,
    )


def _classify_video_bad(
    reasons: List[FramingReasonCode],
    frame_assessments: List[FrameAssessment],
    config: FramingConfig,
) -> Tuple[FramingStatus, str]:
    """Pick video-level status and message for clearly-bad framing."""
    reason_set = set(reasons)
    occupancy_reasons = {
        FramingReasonCode.HIGH_WIDTH_OCCUPANCY,
        FramingReasonCode.HIGH_HEIGHT_OCCUPANCY,
    }

    if FramingReasonCode.EDGE_ADJACENT_MISSING in reason_set:
        cropped_names: List[str] = []
        for fa in frame_assessments:
            cropped_names.extend(fa.metrics.edge_adjacent_missing_names)
        body_part = _describe_cropped_region(cropped_names)
        return (
            FramingStatus.CROPPED,
            f"Your {body_part} is too close to the frame edge. "
            "Move the camera farther back and keep your full riding position visible.",
        )

    if reason_set & occupancy_reasons:
        return (
            FramingStatus.TOO_CLOSE,
            "Camera is too close for accurate bike fit analysis. "
            "Move the camera farther back so your whole body and bike "
            "fit comfortably inside the frame.",
        )

    if FramingReasonCode.LOW_OCCUPANCY in reason_set:
        return (
            FramingStatus.TOO_FAR,
            "The rider appears very small in the frame. "
            "Move the camera closer so the rider fills more of the view.",
        )

    return (
        FramingStatus.LOW_CONFIDENCE,
        "Pose detection confidence is consistently low. Ensure good lighting, "
        "a clear side-on view, and that the full rider is visible.",
    )


def _classify_video_warn(
    reasons: List[FramingReasonCode],
) -> Tuple[FramingStatus, str]:
    """Pick video-level status and message for borderline framing."""
    reason_set = set(reasons)
    occupancy_reasons = {
        FramingReasonCode.HIGH_WIDTH_OCCUPANCY,
        FramingReasonCode.HIGH_HEIGHT_OCCUPANCY,
    }

    if reason_set & occupancy_reasons:
        return (
            FramingStatus.TOO_CLOSE,
            "The rider may be a bit close to the camera. Results are usable "
            "but accuracy improves with more space around the rider.",
        )
    return (
        FramingStatus.LOW_CONFIDENCE,
        "Some pose landmarks have lower confidence than ideal. "
        "Results may be less accurate.",
    )


# ---------------------------------------------------------------------------
# Extension point: perspective distortion (future)
# ---------------------------------------------------------------------------

def assess_perspective_distortion(
    landmarks: List[LandmarkPoint],
    config: FramingConfig = DEFAULT_CONFIG,
) -> Optional[float]:
    """Placeholder for future perspective distortion assessment.

    Could check for suspicious body-segment proportion distortion
    or large frame-to-frame scale instability. Returns None until
    implemented.
    """
    return None
