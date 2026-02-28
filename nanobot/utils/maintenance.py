import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import litellm


def ensure_workspace_structure(workspace: Path) -> None:
    (workspace / "diary").mkdir(parents=True, exist_ok=True)
    task_dir = workspace / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    data_dir = workspace / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    mission = task_dir / "mission.md"
    if not mission.exists():
        mission.write_text("# Missions\n\n", encoding="utf-8")
    coord_log = task_dir / "coord.md"
    if not coord_log.exists():
        coord_log.write_text("# Coordination Log\n\n", encoding="utf-8")
    openrouter_dest = data_dir / "openrouter_models.json"
    if not openrouter_dest.exists():
        candidates = [
            Path("/app/openrouter_models.json"),
            Path(__file__).resolve().parents[1] / "openrouter_models.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    shutil.copyfile(candidate, openrouter_dest)
                    break
                except Exception:
                    continue


def ensure_system_jobs(cron_service, workspace: Path) -> None:
    from nanobot.cron.types import CronSchedule
    existing = {j.name for j in cron_service.list_jobs(include_disabled=True)}

    if "update-openrouter-models" not in existing:
        cron_service.add_job(
            name="update-openrouter-models",
            schedule=CronSchedule(kind="cron", expr="0 3 * * 0"),
            message="system:update_openrouter_models",
            deliver=False,
        )

    if "daily-diary" not in existing:
        cron_service.add_job(
            name="daily-diary",
            schedule=CronSchedule(kind="cron", expr="0 0 * * *"),
            message=(
                "Write a daily diary entry for today from your perspective. "
                "Append to workspace/diary/YYYY-MM.md under a section header "
                "'## YYYY-MM-DD - Weekday'. Keep it concise and factual."
            ),
            deliver=False,
        )

    if "daily-task-review" not in existing:
        cron_service.add_job(
            name="daily-task-review",
            schedule=CronSchedule(kind="cron", expr="0 9 * * *"),
            message=(
                "Review workspace/task/mission.md and task-*.md files. "
                "Summarize progress, update statuses, and decide next actions. "
                "If a task is stalled, note blockers."
            ),
            deliver=False,
        )


def handle_system_event(message: str, workspace: Path) -> str:
    if message == "update_openrouter_models":
        updated = update_openrouter_models(workspace, allow_fetch=True)
        return "openrouter models updated" if updated else "openrouter models update skipped"
    return "unknown system event"


def update_openrouter_models(workspace: Path, allow_fetch: bool) -> bool:
    data = load_openrouter_models(workspace, allow_fetch=allow_fetch)
    if not data:
        return False
    apply_openrouter_pricing(data)
    return True


def load_openrouter_models(workspace: Path, allow_fetch: bool, max_age_days: int = 7) -> dict[str, Any] | None:
    cache_path = workspace / "data" / "openrouter_models_cache.json"
    fallback_path = workspace / "data" / "openrouter_models.json"

    cache = _read_json(cache_path)
    if cache:
        fetched_at = cache.get("fetched_at")
        if fetched_at:
            try:
                ts = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
                if datetime.utcnow() - ts <= timedelta(days=max_age_days):
                    return cache
            except Exception:
                pass

    if allow_fetch:
        data = _fetch_openrouter()
        if data:
            _write_cache(cache_path, data, source="api")
            _write_models(fallback_path, data)
            return data

    fallback = _read_json(fallback_path)
    if not fallback:
        fallback = _read_json(Path(__file__).resolve().parents[1] / "openrouter_models.json")
    if fallback:
        _write_cache(cache_path, fallback, source="fallback")
        return fallback
    return None


def apply_openrouter_pricing(cache: dict[str, Any]) -> None:
    data = cache.get("data") if isinstance(cache, dict) else None
    if not isinstance(data, list):
        return
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        pricing = item.get("pricing") or {}
        if not model_id or not isinstance(pricing, dict):
            continue
        prompt = _to_float(pricing.get("prompt"))
        completion = _to_float(pricing.get("completion"))
        if prompt is None and completion is None:
            continue
        prompt_val = prompt or 0.0
        completion_val = completion or 0.0
        for key in (model_id, f"openrouter/{model_id}"):
            litellm.model_cost[key] = {
                "input_cost_per_token": prompt_val,
                "output_cost_per_token": completion_val,
            }


def _fetch_openrouter() -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get("https://openrouter.ai/api/v1/models")
            if resp.status_code != 200:
                return None
            data = resp.json()
            if isinstance(data, dict) and "data" in data:
                return data
    except Exception:
        return None
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: Path, data: dict[str, Any], source: str) -> None:
    payload = {
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "source": source,
        **data,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_models(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_float(val: Any) -> float | None:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except Exception:
            return None
    return None
