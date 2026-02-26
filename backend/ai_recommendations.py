import json
import os
from typing import Dict, List, Optional

from openai import AsyncOpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """\
You are an expert bicycle fitter with 20+ years of experience fitting riders \
of all levels — from weekend warriors to professional cyclists. You combine \
biomechanical knowledge, practical workshop experience, and an understanding \
of rider comfort and injury prevention.

You will receive a structured analysis of a rider's position captured from \
video. Your job is to synthesize *all* the data into a clear, actionable \
summary that a rider can immediately use to improve their fit.

Guidelines:
- Be conversational but authoritative. Write like you're talking to the rider \
  in your shop.
- Lead with the most impactful finding — the single change that will make the \
  biggest difference.
- Explain *why* each adjustment matters (comfort, power, injury risk).
- When angles interact (e.g., saddle height affects both knee AND hip), \
  explain the trade-off.
- Account for the rider's stated goal and mobility. A comfort-focused rider \
  with tight hamstrings gets different advice than an aero-chasing racer.
- If the rider included notes about pain or discomfort, address those directly.
- End with a clear 1-2-3 priority list of changes to try, with approximate \
  adjustment amounts where applicable.
- Keep the total response under 300 words. Riders want clarity, not an essay.
- Do NOT repeat raw numbers excessively. Reference them once when relevant.
- Use markdown formatting (bold, bullets) for readability.\
"""


def _build_user_prompt(
    angles: Dict[str, Optional[float]],
    fit_windows: Dict[str, dict],
    stroke_samples: Dict[str, dict],
    recommended_actions: List[dict],
    bike_type: str,
    bike_type_label: str,
    goal: str,
    mobility: Dict[str, float],
    notes: str,
) -> str:
    sections = []

    sections.append(f"**Bike type:** {bike_type_label} ({bike_type})")
    sections.append(f"**Riding goal:** {goal}")

    if mobility:
        mob_str = ", ".join(f"{k}: {v}/10" for k, v in mobility.items())
        sections.append(f"**Mobility scores:** {mob_str}")

    if notes:
        sections.append(f"**Rider notes:** {notes}")

    sections.append("\n## Measured Angles (best frame — bottom of pedal stroke)")
    for name, value in angles.items():
        label = name.replace("_angle_deg", "").capitalize()
        sections.append(f"- {label}: {value:.1f}°" if value is not None else f"- {label}: not measured")

    sections.append("\n## Fit Windows")
    for metric, window in fit_windows.items():
        status_emoji = {"In Range": "✅", "Slightly Off": "⚠️", "Off": "❌"}.get(window["status"], "")
        sections.append(
            f"- {metric.capitalize()}: {window['measured']}° "
            f"(target {window['target_min']}–{window['target_max']}°) "
            f"→ {status_emoji} {window['status']}"
        )

    if stroke_samples:
        sections.append("\n## Stroke Positions")
        for position, sample_angles in stroke_samples.items():
            knee = sample_angles.get("knee_angle_deg")
            hip = sample_angles.get("hip_angle_deg")
            if knee is not None:
                sections.append(f"- {position.capitalize()}: knee {knee:.1f}°, hip {hip:.1f}°")

    if recommended_actions:
        sections.append("\n## Rule-Based Actions (for reference)")
        for action in recommended_actions:
            sections.append(f"- **{action['title']}**: {action['change']} — {action['reason']}")

    sections.append(
        "\nPlease provide your expert bike fit analysis and prioritized recommendations."
    )

    return "\n".join(sections)


async def generate_ai_summary(
    angles: Dict[str, Optional[float]],
    fit_windows: Dict[str, dict],
    stroke_samples: Dict[str, dict],
    recommended_actions: List[dict],
    bike_type: str,
    bike_type_label: str,
    goal: str,
    mobility: Dict[str, float],
    notes: str,
) -> Optional[str]:
    """Call OpenAI to generate an expert bike fit summary.

    Returns the AI-generated markdown string, or None if the API key
    is not configured or the call fails.
    """
    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    user_prompt = _build_user_prompt(
        angles=angles,
        fit_windows=fit_windows,
        stroke_samples=stroke_samples,
        recommended_actions=recommended_actions,
        bike_type=bike_type,
        bike_type_label=bike_type_label,
        goal=goal,
        mobility=mobility,
        notes=notes,
    )

    try:
        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=600,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"AI recommendation generation failed: {e}")
        return None
