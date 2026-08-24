"""Fail-fast readiness checks for Dataset Builder runs."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from portrait_core.models import ModelManager


@dataclass(frozen=True)
class PreflightError(RuntimeError):
    report: dict[str, Any]

    def __str__(self) -> str:
        messages = [c["message"] for c in self.report["checks"] if c["status"] == "error"]
        return "ORION не готов к запуску: " + "; ".join(messages)


def _check(check_id: str, status: str, message: str, **details) -> dict[str, Any]:
    return {"id": check_id, "status": status, "message": message, "details": details}


def run_preflight(input_path: str, output_dir: str, *, backend: str = "mediapipe",
                  model_path: str | None = None, topology_path: str | None = None,
                  initialize_backend: bool = True) -> dict[str, Any]:
    checks = []
    checks.append(_check("python.portrait_core", "pass", "portrait_core импортирован"))
    for module in ("cv2", backend if backend != "onnx" else "onnxruntime"):
        available = importlib.util.find_spec(module) is not None
        checks.append(_check(f"dependency.{module}", "pass" if available else "error",
                             f"{module} доступен" if available else f"Не установлена зависимость: {module}"))
    manager = ModelManager()
    resolved = manager.resolve_backend(backend, explicit_path=model_path)
    validated = manager.validate(resolved, initialize_backend=initialize_backend, topology_path=topology_path)
    checks.append(_check(f"model.{resolved.model_id}", "pass" if validated.valid else "error",
                         validated.message or "Ошибка модели", requested_path=model_path,
                         resolved_path=str(validated.path), resolution_source=validated.source))
    source_is_url = input_path.lower().startswith(("http://", "https://"))
    if source_is_url:
        available = importlib.util.find_spec("yt_dlp") is not None or shutil.which("yt-dlp") is not None
        checks.append(_check("dependency.yt_dlp", "pass" if available else "error",
                             "yt-dlp доступен" if available else "yt-dlp недоступен для URL"))
    else:
        accessible = Path(input_path).exists()
        checks.append(_check("input.accessible", "pass" if accessible else "error",
                             "Источник доступен" if accessible else f"Источник не найден: {input_path}"))
    for tool in ("ffmpeg", "ffprobe"):
        available = shutil.which(tool) is not None
        checks.append(_check(f"dependency.{tool}", "pass" if available else "warning",
                             f"{tool} доступен" if available else f"{tool} недоступен; будет использован fallback"))
    destination = Path(output_dir).expanduser().resolve(strict=False)
    parent = destination if destination.exists() else destination.parent
    writable = parent.exists() and parent.is_dir()
    checks.append(_check("output.writable", "pass" if writable else "error",
                         "Каталог результата доступен" if writable else f"Каталог результата недоступен: {parent}"))
    status = "failed" if any(c["status"] == "error" for c in checks) else "ready"
    return {"schema": "orion.preflight.v1", "status": status, "checks": checks,
            "analysis_backend": {"name": backend, "model_id": validated.model_id,
                                 "model_path": validated.to_dict(project_root=manager.project_root)["path"],
                                 "model_resolution_source": validated.source,
                                 "model_validated": validated.valid,
                                 "model": validated.to_dict(project_root=manager.project_root)}}


def require_preflight(*args, **kwargs) -> dict[str, Any]:
    report = run_preflight(*args, **kwargs)
    if report["status"] != "ready":
        raise PreflightError(report)
    return report
