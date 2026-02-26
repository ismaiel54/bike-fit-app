# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Bike Fit Analyzer — a single-service Python/FastAPI web app that analyzes cycling posture from uploaded video using MediaPipe pose estimation and OpenCV. No database, no separate frontend build, no Docker.

### Key caveats

- **mediapipe version**: The code uses the legacy `mp.solutions.pose` API. You must install `mediapipe<0.10.15` (e.g. 0.10.14). Versions >= 0.10.15 removed `mp.solutions` and will crash on import.
- **Working directory**: The FastAPI app must be started from `/workspace/backend/` because it uses relative paths for `templates/`, `outputs/`, and `static/`.
- **PATH**: `uvicorn` installs to `~/.local/bin`. Ensure `export PATH="$HOME/.local/bin:$PATH"` is set before running.

### Running the dev server

```bash
cd /workspace/backend
export PATH="$HOME/.local/bin:$PATH"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The app serves at `http://localhost:8000`. The `--reload` flag enables hot-reloading on code changes.

### Linting / Testing

- No linting tools or test framework are configured in this repository.
- No automated tests exist. Manual testing is done via the web UI or `curl` against `POST /analyze-video`.

### API endpoints

- `GET /` — serves the single-page UI (Jinja2 template)
- `POST /analyze-video` — accepts multipart form with `file` (MP4/MOV), `bike_type`, `goal`, `mobility` (JSON), `notes`; returns JSON with angles, fit windows, recommendations, and annotated media URLs.
