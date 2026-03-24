import json
import os
import shutil
import uuid
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import Dict, Literal, Optional, Tuple

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from framing_assessment import (
    DEFAULT_CONFIG as FRAMING_CONFIG,
    FrameAssessment,
    LandmarkPoint,
    VideoAssessment,
    aggregate_video_framing,
    assess_frame_framing,
)
from pose_analysis import (
    analyze_pose_from_frame,
    compute_event_median_metrics,
    compute_fit_windows,
    compute_stroke_samples,
    detect_pedal_events,
    draw_pose_overlay,
    generate_annotated_video,
    generate_bikefit_recommendations,
    generate_recommended_actions,
    generate_report,
    get_target_ranges,
    select_clip_analysis_side,
)

# Map frontend bike types to internal codes
BIKE_TYPE_MAP = {
    "Road": "road",
    "Gravel-CX": "gravel",
    "MTB": "mtb",
    "Triathlon": "tt",
    "Time Trial": "tt",
    "Spin-Exercise": "spin",
}

BikeType = Literal["tt", "road", "gravel", "mtb", "spin"]
GoalType = Literal["Comfort", "Balanced", "Aero-Performance"]

BIKE_TYPE_CONFIG: Dict[BikeType, dict] = {
    "tt": {
        "label": "TT / Triathlon bike",
        "knee": {
            "optimal": (140.0, 148.0),
            "neutral": (138.0, 150.0),
        },
        "hip": {
            "optimal": (50.0, 58.0),
            "neutral": (48.0, 62.0),
        },
        "foot": {
            "neutral": (85.0, 100.0),
            "ok": (82.0, 102.0),
        },
        "summary_focus": "Aggressive aero fit prioritising low drag while keeping hip angle open enough for sustainable power.",
    },
    "road": {
        "label": "Road bike",
        "knee": {
            "optimal": (140.0, 150.0),
            "neutral": (138.0, 152.0),
        },
        "hip": {
            "optimal": (50.0, 60.0),
            "neutral": (45.0, 65.0),
        },
        "foot": {
            "neutral": (90.0, 105.0),
            "ok": (85.0, 110.0),
        },
        "summary_focus": "Balanced fit for power, comfort, and long-duration riding.",
    },
    "gravel": {
        "label": "Gravel bike",
        "knee": {
            "optimal": (140.0, 150.0),
            "neutral": (138.0, 152.0),
        },
        "hip": {
            "optimal": (55.0, 65.0),
            "neutral": (50.0, 70.0),
        },
        "foot": {
            "neutral": (95.0, 110.0),
            "ok": (90.0, 112.0),
        },
        "summary_focus": "Slightly more upright fit for stability, comfort and mixed terrain.",
    },
    "mtb": {
        "label": "Mountain bike",
        "knee": {
            "optimal": (140.0, 150.0),
            "neutral": (138.0, 152.0),
        },
        "hip": {
            "optimal": (60.0, 75.0),
            "neutral": (55.0, 80.0),
        },
        "foot": {
            "neutral": (100.0, 115.0),
            "ok": (95.0, 118.0),
        },
        "summary_focus": "Upright, controlled fit for technical terrain, drops and variable body positions.",
    },
    "spin": {
        "label": "Spin / Indoor bike",
        "knee": {
            "optimal": (140.0, 150.0),
            "neutral": (138.0, 152.0),
        },
        "hip": {
            "optimal": (55.0, 65.0),
            "neutral": (50.0, 70.0),
        },
        "foot": {
            "neutral": (90.0, 105.0),
            "ok": (85.0, 110.0),
        },
        "summary_focus": "Comfort-first indoor setup for sustainable cadence and general fitness.",
    },
}

app = FastAPI(title="Bike Fit API", version="0.1.0")

# Create directories if they don't exist
if not os.path.exists("outputs"):
    os.makedirs("outputs")
if not os.path.exists("static"):
    os.makedirs("static")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main bike-fit UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(...),
    bike_type: str = Form("Road"),
    goal: str = Form("Balanced"),
    mobility: str = Form("{}"),  # JSON string
    notes: str = Form(""),
) -> Dict[str, object]:
    """
    Accepts a video upload, finds the best frame (where leg is most extended),
    runs pose analysis, generates an annotated image, and returns bike-fit recommendations.
    """
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in {".mp4", ".mov"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .mp4 or .mov")

    temp_path: Optional[str] = None
    try:
        with NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not read the uploaded video")

        # Reset smoothing state for new video
        from pose_analysis import _smoothing_state
        _smoothing_state["foot_angle_ema"] = None
        _smoothing_state["torso_angle_ema"] = None
        _smoothing_state["torso_debug"] = None
        _smoothing_state["display_rotation_deg"] = 0  # Reset rotation tracking

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise HTTPException(status_code=400, detail="Video has no readable frames")

        # Parse mobility JSON
        try:
            mobility_dict = json.loads(mobility) if mobility else {}
        except json.JSONDecodeError:
            mobility_dict = {}

        # Map frontend bike type to internal code
        internal_bike_type = BIKE_TYPE_MAP.get(bike_type, "road")
        if internal_bike_type not in BIKE_TYPE_CONFIG:
            internal_bike_type = "road"

        # Sample frames throughout the video to extract stroke positions
        sample_indices = [
            int(i * frame_count / 20) for i in range(20)
        ]  # Sample 20 frames for better stroke detection

        frame_results = []
        frame_rgb_samples: Dict[int, object] = {}
        framing_frame_assessments: list[FrameAssessment] = []
        min_knee_angle = float("inf")
        max_knee_angle = -1
        min_knee_idx = 0
        max_knee_idx = 0

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame_bgr = cap.read()
            if not success or frame_bgr is None:
                continue

            # Normalize frame orientation before processing
            # Ensure horizontal is true horizontal (handle rotation if needed)
            from pose_analysis import _normalize_frame_orientation
            frame_bgr_normalized, rotation_deg = _normalize_frame_orientation(frame_bgr.copy())
            
            frame_rgb = cv2.cvtColor(frame_bgr_normalized, cv2.COLOR_BGR2RGB)
            pose_result = analyze_pose_from_frame(frame_rgb)

            if pose_result.get("pose_detected"):
                knee_angle = pose_result.get("knee_angle_deg", 0)
                frame_results.append((idx, frame_bgr_normalized.copy(), pose_result, knee_angle))
                frame_rgb_samples[idx] = frame_rgb

                # Track min/max knee angles for stroke positions
                if knee_angle < min_knee_angle:
                    min_knee_angle = knee_angle
                    min_knee_idx = idx
                if knee_angle > max_knee_angle:
                    max_knee_angle = knee_angle
                    max_knee_idx = idx

                # Framing assessment: convert landmark visibility to LandmarkPoints
                vis_data = pose_result.get("landmark_visibility", {})
                framing_landmarks = [
                    LandmarkPoint(
                        name=name,
                        x=info["x"],
                        y=info["y"],
                        confidence=info["confidence"],
                    )
                    for name, info in vis_data.items()
                ]
                if framing_landmarks:
                    framing_frame_assessments.append(
                        assess_frame_framing(framing_landmarks, FRAMING_CONFIG)
                    )

        cap.release()

        # Aggregate framing assessment across all sampled frames
        framing_result: VideoAssessment = aggregate_video_framing(
            framing_frame_assessments, FRAMING_CONFIG
        )

        if not frame_results:
            raise HTTPException(
                status_code=422, detail="Pose not detected in any sampled frame. Ensure the rider is visible from the left side."
            )

        # Lock one side for the full clip, then re-analyze sampled frames using that side.
        locked_side = select_clip_analysis_side([r[2] for r in frame_results], first_n_valid_frames=20)

        _smoothing_state["foot_angle_ema"] = None
        _smoothing_state["torso_angle_ema"] = None
        _smoothing_state["torso_debug"] = None

        locked_frame_results = []
        frame_metrics = []
        for idx, frame_bgr, _, _ in frame_results:
            frame_rgb = frame_rgb_samples[idx]
            locked_result = analyze_pose_from_frame(frame_rgb, locked_side=locked_side)
            if not locked_result.get("pose_detected"):
                continue
            knee_val = locked_result.get("knee_angle_deg", 0)
            locked_frame_results.append((idx, frame_bgr, locked_result, knee_val))
            landmarks_norm = locked_result.get("landmarks_norm", {})
            frame_metrics.append(
                {
                    "frame_index": idx,
                    "ankle_norm": landmarks_norm.get("ankle"),
                    "knee_angle_deg": locked_result.get("knee_angle_deg"),
                    "hip_angle_deg": locked_result.get("hip_angle_deg"),
                    "foot_angle_deg": locked_result.get("foot_angle_deg"),
                    "torso_angle_deg": locked_result.get("torso_angle_deg"),
                    "elbow_angle_deg": locked_result.get("elbow_angle_deg"),
                }
            )

        if not locked_frame_results:
            raise HTTPException(
                status_code=422,
                detail="Pose not detected in any sampled frame after side lock.",
            )

        # Recompute stroke extrema from the locked, standardized metric stream.
        min_knee_idx = min(
            locked_frame_results,
            key=lambda row: row[3] if row[3] is not None else float("inf"),
        )[0]
        max_knee_idx = max(
            locked_frame_results,
            key=lambda row: row[3] if row[3] is not None else float("-inf"),
        )[0]

        # Event-based clip metrics (median aggregation)
        events = detect_pedal_events(frame_metrics)
        clip_metrics = compute_event_median_metrics(frame_metrics, events)

        # Representative frame for output image: first bottom-event frame if available.
        if events.get("bottom"):
            event_i = events["bottom"][0]
            bottom_frame_idx = frame_metrics[event_i]["frame_index"]
            best_frame_index = int(bottom_frame_idx)
        else:
            best_frame_index = max_knee_idx

        best_frame_bgr = None
        best_pose_result = None
        for idx, frame_bgr, pose_result, _ in locked_frame_results:
            if idx == best_frame_index:
                best_frame_bgr = frame_bgr
                best_pose_result = pose_result
                break

        # Compute stroke samples (top, mid, bottom)
        stroke_samples = compute_stroke_samples(
            locked_frame_results, int(min_knee_idx), int(max_knee_idx), frame_count
        )

        if best_pose_result is None or not best_pose_result.get("pose_detected"):
            raise HTTPException(
                status_code=422, detail="Pose not detected in any sampled frame. Ensure the rider is visible from the left side."
            )

        knee_angle = clip_metrics.get("knee_angle_deg") or best_pose_result["knee_angle_deg"]
        hip_closed = clip_metrics.get("hip_closed_deg")
        hip_open = clip_metrics.get("hip_open_deg")
        # Backward-compatible primary hip output: prefer hip_closed, fallback to representative frame hip.
        hip_angle = hip_closed if hip_closed is not None else best_pose_result["hip_angle_deg"]
        foot_angle = clip_metrics.get("foot_angle_deg")
        if foot_angle is None:
            foot_angle = best_pose_result.get("foot_angle_deg")
        torso_angle = clip_metrics.get("torso_angle_deg")
        if torso_angle is None:
            torso_angle = best_pose_result.get("torso_angle_deg")
        elbow_angle = clip_metrics.get("elbow_angle_deg")
        if elbow_angle is None:
            elbow_angle = best_pose_result.get("elbow_angle_deg")
        landmarks_px = best_pose_result["landmarks_px"]

        bike_config = BIKE_TYPE_CONFIG[internal_bike_type]

        # Get target ranges based on bike type and goal
        target_ranges = get_target_ranges(internal_bike_type, goal, mobility_dict)

        # Generate recommendations and report with bike type
        recommendations = generate_bikefit_recommendations(
            {
                "knee_angle_deg": knee_angle,
                "hip_angle_deg": hip_angle,
                "foot_angle_deg": foot_angle,
                "torso_angle_deg": torso_angle,
                "elbow_angle_deg": elbow_angle,
            },
            bike_type=internal_bike_type,
            bike_config=bike_config,
            goal=goal,
            mobility=mobility_dict,
        )
        report = generate_report(knee_angle, hip_angle, recommendations, bike_config)

        # Compute fit windows
        angles_dict = {
            "knee_angle_deg": knee_angle,
            "hip_angle_deg": hip_angle,
            "foot_angle_deg": foot_angle,
            "torso_angle_deg": torso_angle,
            "elbow_angle_deg": elbow_angle,
        }
        fit_windows = compute_fit_windows(angles_dict, target_ranges)

        # Generate recommended actions
        recommended_actions = generate_recommended_actions(fit_windows, internal_bike_type, goal)

        # Generate annotated video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        video_filename = f"annotated_{timestamp}_{unique_id}.mp4"
        video_output_path = os.path.join("outputs", video_filename)

        try:
            generate_annotated_video(temp_path, video_output_path, sample_every_n_frames=1)
        except Exception as e:
            # If video generation fails, still return the image analysis
            print(f"Warning: Video generation failed: {e}")

        # Also save annotated image for reference
        best_frame_rgb = cv2.cvtColor(best_frame_bgr, cv2.COLOR_BGR2RGB)
        annotated_frame = draw_pose_overlay(
            best_frame_rgb,
            landmarks_px,
            knee_angle,
            hip_angle,
            foot_angle_deg=foot_angle,
            torso_angle_deg=torso_angle,
            elbow_angle_deg=elbow_angle,
        )
        image_filename = f"annotated_{timestamp}_{unique_id}.png"
        image_output_path = os.path.join("outputs", image_filename)
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_output_path, annotated_frame_bgr)

        # Construct URLs
        base_url = "http://127.0.0.1:8000"
        annotated_video_url = f"{base_url}/outputs/{video_filename}"
        annotated_image_url = f"{base_url}/outputs/{image_filename}"

        return {
            "pose_detected": True,
            "frame_index_used": best_frame_index,
            "bike_type": bike_type,
            "bike_type_internal": internal_bike_type,
            "bike_type_label": bike_config["label"],
            "goal": goal,
            "mobility": mobility_dict,
            "notes": notes,
            "angles": {
                "knee_angle_deg": knee_angle,
                "hip_angle_deg": hip_angle,
                "hip_closed_deg": hip_closed,
                "hip_open_deg": hip_open,
                "foot_angle_deg": foot_angle,
                "torso_angle_deg": torso_angle,
                "elbow_angle_deg": elbow_angle,
            },
            "angle_details": {
                "raw_from_representative_frame": best_pose_result.get("raw_angles_deg", {}),
                "display_from_representative_frame": {
                    "knee_angle_deg": best_pose_result.get("knee_angle_deg"),
                    "hip_angle_deg": best_pose_result.get("hip_angle_deg"),
                    "foot_angle_deg": best_pose_result.get("foot_angle_deg"),
                    "torso_angle_deg": best_pose_result.get("torso_angle_deg"),
                    "elbow_angle_deg": best_pose_result.get("elbow_angle_deg"),
                },
                "display_convention": best_pose_result.get("display_convention", {}),
            },
            "fit_windows": fit_windows,
            "stroke_samples": stroke_samples,
            "recommended_actions": recommended_actions,
            "recommendations": recommendations,
            "report": report,
            "annotated_image_url": annotated_image_url,
            "annotated_video_url": annotated_video_url,
            "framing_assessment": framing_result.to_dict(),
            "event_detection": events,
            "aggregation_method": "event_median_with_representative_fallback",
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
