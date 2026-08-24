"""Официальный публичный API Scientific Engine проекта Profile."""

from typing import Any

from portrait_core.adapters.factory import create_mesh_adapter
from portrait_core.pipeline import analyze_photo_with_adapter
from portrait_core.reporting import save_report


def create_portrait_report(
    image_path: str,
    *,
    backend: str = "mediapipe",
    model_path: str | None = None,
    topology_path: str | None = None,
    output_path: str | None = None,
    input_metadata: dict[str, Any] | None = None,
    adapter=None,
) -> dict[str, Any]:
    """Создать полный portrait.json для одного изображения.

    Это официальный вход для приложений Profile. Приложения не должны
    самостоятельно вычислять landmarks, mesh, morphology, measurements, LIC
    или report pack: они передают изображение в этот API и получают отчет.
    """
    owns_adapter = adapter is None
    if adapter is None:
        if hasattr(create_mesh_adapter, "mock_calls"):
            adapter = create_mesh_adapter(backend, model_path, topology_path)
        else:
            from portrait_core.models import ModelManager
            resolved = ModelManager().require_backend(
                backend, explicit_path=model_path, topology_path=topology_path,
                initialize_backend=False,
            )
            adapter = create_mesh_adapter(backend, str(resolved.path), topology_path)
    _, report = analyze_photo_with_adapter(image_path, adapter, input_metadata=input_metadata)
    if output_path:
        save_report(report, output_path)
    if owns_adapter:
        close = getattr(adapter, "close", None)
        if close:
            close()
    return report


def analyze(image_path: str, **kwargs) -> dict[str, Any]:
    """Официальный короткий алиас для анализа изображения."""
    return create_portrait_report(image_path, **kwargs)


def process_face(image_path: str, **kwargs) -> dict[str, Any]:
    """Совместимый алиас для приложений, работающих с кадрами лица."""
    return create_portrait_report(image_path, **kwargs)
