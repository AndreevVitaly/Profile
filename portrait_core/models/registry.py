"""Declarative registry of scientific models used by ORION."""

MODEL_REGISTRY = {
    "mediapipe_face_landmarker": {
        "backend": "mediapipe",
        "filename": "face_landmarker.task",
        "environment": "ORION_FACE_LANDMARKER_MODEL",
        "required": True,
        "type": "mediapipe_task",
        "extensions": (".task",),
        "download_url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    },
}

BACKEND_MODELS = {spec["backend"]: model_id for model_id, spec in MODEL_REGISTRY.items()}
