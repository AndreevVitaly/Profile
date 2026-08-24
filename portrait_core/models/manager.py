"""Portable model resolution, validation and provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .registry import BACKEND_MODELS, MODEL_REGISTRY


class ModelValidationError(RuntimeError):
    """A required model or its backend cannot be initialized."""


@dataclass(frozen=True)
class ResolvedModel:
    model_id: str
    path: Path
    source: str
    exists: bool
    valid: bool = False
    message: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    backend_version: str | None = None

    def to_dict(self, *, project_root: Path | None = None) -> dict[str, Any]:
        result = asdict(self)
        path = self.path
        if project_root:
            try:
                path = path.relative_to(project_root)
            except ValueError:
                pass
        result["path"] = path.as_posix()
        result["validation_status"] = "passed" if self.valid else "failed"
        return result


class ModelManager:
    def __init__(self, *, config: dict[str, str] | None = None, user_models_dir: str | Path | None = None):
        self.config = config or {}
        self.project_root = Path(__file__).resolve().parents[2]
        self.project_models_dir = Path(__file__).resolve().parent
        self.user_models_dir = Path(user_models_dir) if user_models_dir else Path.home() / ".orion" / "models"

    def model_id_for_backend(self, backend: str) -> str:
        try:
            return BACKEND_MODELS[backend]
        except KeyError as error:
            raise ModelValidationError(f"Для backend '{backend}' модель не зарегистрирована") from error

    def resolve(self, model_id: str, *, explicit_path: str | Path | None = None) -> ResolvedModel:
        try:
            spec = MODEL_REGISTRY[model_id]
        except KeyError as error:
            raise ModelValidationError(f"Неизвестная модель ORION: {model_id}") from error
        candidates: list[tuple[str, Path]] = []
        if explicit_path:
            candidates.append(("explicit", Path(explicit_path).expanduser()))
        configured = self.config.get(model_id)
        if configured:
            candidates.append(("config", Path(configured).expanduser()))
        environment = os.environ.get(spec["environment"])
        if environment:
            candidates.append(("environment", Path(environment).expanduser()))
        candidates.extend([
            ("project_default", self.project_models_dir / spec["filename"]),
            ("user_default", self.user_models_dir / spec["filename"]),
        ])
        if explicit_path or configured or environment:
            source, path = candidates[0]
        else:
            source, path = candidates[0]
            for candidate_source, candidate in candidates:
                if candidate.is_file():
                    source, path = candidate_source, candidate
                    break
        path = path.resolve(strict=False)
        return ResolvedModel(model_id, path, source, path.exists())

    def resolve_backend(self, backend: str, *, explicit_path: str | Path | None = None) -> ResolvedModel:
        if backend == "onnx":
            if not explicit_path:
                raise ModelValidationError("Для ONNX backend требуется явный путь к модели")
            path = Path(explicit_path).expanduser().resolve(strict=False)
            return ResolvedModel("onnx_custom", path, "explicit", path.exists())
        return self.resolve(self.model_id_for_backend(backend), explicit_path=explicit_path)

    def validate(self, resolved: ResolvedModel, *, initialize_backend: bool = True, topology_path: str | None = None) -> ResolvedModel:
        path = resolved.path
        error = None
        if not path.exists():
            error = f"Обязательная модель не найдена: {path}"
        elif not path.is_file():
            error = f"Путь модели не является файлом: {path}"
        else:
            try:
                with path.open("rb") as stream:
                    first = stream.read(1)
                if not first:
                    error = f"Файл модели пуст: {path}"
            except OSError as exc:
                error = f"Файл модели недоступен для чтения: {path}: {exc}"
        expected = (".onnx",) if resolved.model_id == "onnx_custom" else MODEL_REGISTRY[resolved.model_id]["extensions"]
        if error is None and path.suffix.lower() not in expected:
            error = f"Неподдерживаемый тип модели: {path.suffix or '<без расширения>'}"
        if error is None and initialize_backend:
            try:
                from portrait_core.adapters.factory import create_mesh_adapter
                adapter = create_mesh_adapter("onnx" if resolved.model_id == "onnx_custom" else "mediapipe", str(path), topology_path)
                prepare = getattr(adapter, "prepare", None)
                if prepare:
                    prepare()
                close = getattr(adapter, "close", None)
                if close:
                    close()
            except Exception as exc:  # backend libraries expose heterogeneous errors
                error = f"Backend не смог инициализировать модель: {exc}"
        if error:
            return ResolvedModel(**{**asdict(resolved), "valid": False, "message": error})
        content = path.read_bytes()
        package = "onnxruntime" if resolved.model_id == "onnx_custom" else "mediapipe"
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        return ResolvedModel(**{**asdict(resolved), "valid": True, "message": "Модель готова", "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "backend_version": version})

    def install(self, model_id: str, *, destination: str | Path | None = None) -> ResolvedModel:
        spec = MODEL_REGISTRY[model_id]
        target = Path(destination) if destination else self.project_models_dir / spec["filename"]
        target = target.expanduser().resolve(strict=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=".download") as stream:
                temporary = Path(stream.name)
                with urllib.request.urlopen(spec["download_url"], timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if digest != spec["sha256"]:
                raise ModelValidationError(f"Контрольная сумма модели не совпадает: {digest}")
            temporary.replace(target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return self.validate(ResolvedModel(model_id, target, "installed", True), initialize_backend=True)

    def require_backend(self, backend: str, *, explicit_path: str | Path | None = None, topology_path: str | None = None, initialize_backend: bool = True) -> ResolvedModel:
        result = self.validate(self.resolve_backend(backend, explicit_path=explicit_path), initialize_backend=initialize_backend, topology_path=topology_path)
        if not result.valid:
            raise ModelValidationError(result.message or "Модель не готова")
        return result
