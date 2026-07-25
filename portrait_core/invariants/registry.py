"""Central registry for geometric ratio candidates.

The ratios defined here are computable candidates. A stable value in one
dataset is not a universal validated face invariant.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RatioDefinition:
    name: str
    numerator: str
    denominator: str
    category: str
    description: str
    unit: str = "px"


INVARIANT_SCHEMA = {
    "name": "profile-geometric-invariants",
    "version": "1.0",
}


STATS_SCHEMA = {
    "name": "profile-geometric-invariant-stats",
    "version": "1.0",
}


INVARIANT_DEFINITIONS: tuple[RatioDefinition, ...] = (
    RatioDefinition("face_height_face_width", "face_height", "face_width", "face", "Face height relative to face width"),
    RatioDefinition(
        "ipd_face_width",
        "interpupillary_distance",
        "face_width",
        "eyes",
        "Interpupillary distance relative to face width",
    ),
    RatioDefinition("nose_length_face_height", "nose_length", "face_height", "nose", "Nose length relative to face height"),
    RatioDefinition("nose_width_face_width", "nose_width", "face_width", "nose", "Nose width relative to face width"),
    RatioDefinition("mouth_width_face_width", "mouth_width", "face_width", "mouth", "Mouth width relative to face width"),
    RatioDefinition("jaw_width_face_width", "jaw_width", "face_width", "jaw", "Jaw width relative to face width"),
    RatioDefinition("forehead_height_face_height", "forehead_height", "face_height", "forehead", "Forehead height relative to face height"),
    RatioDefinition("lower_face_height_face_height", "lower_face_height", "face_height", "face", "Lower face height relative to face height"),
    RatioDefinition("eye_width_left_face_width", "eye_width_left", "face_width", "eyes", "Left eye width relative to face width"),
    RatioDefinition("eye_width_right_face_width", "eye_width_right", "face_width", "eyes", "Right eye width relative to face width"),
)


MEASUREMENT_ALIASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "face_width": (("face", "face_width"), ("face_width",)),
    "face_height": (("face", "face_height"), ("face_height",)),
    "interpupillary_distance": (
        ("eyes", "interpupillary_distance"),
        ("eyes", "ipd"),
        ("eyes", "eye_distance"),
        ("interpupillary_distance",),
        ("ipd",),
        ("interocular_distance",),
    ),
    "eye_width_left": (("eyes", "eye_width_left"), ("eyes", "left_eye_width"), ("eye_width_left",)),
    "eye_width_right": (("eyes", "eye_width_right"), ("eyes", "right_eye_width"), ("eye_width_right",)),
    "nose_length": (("nose", "nose_length"), ("nose_length",)),
    "nose_width": (("nose", "nose_width"), ("nose_width",)),
    "mouth_width": (("mouth", "mouth_width"), ("mouth_width",)),
    "jaw_width": (("jaw", "jaw_width"), ("jaw_width",)),
    "forehead_height": (("forehead", "forehead_height"), ("forehead_height",)),
    "lower_face_height": (("face", "lower_face_height"), ("jaw", "lower_face_height"), ("lower_face_height",)),
}


STABILITY_THRESHOLDS = {
    "minimum_count": 3,
    "excellent_cv": 0.05,
    "stable_cv": 0.10,
    "moderate_cv": 0.20,
}
