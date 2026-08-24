"""Объективная оценка пригодности фотографии для измерений."""

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from portrait_core.config import QUALITY_THRESHOLDS
from portrait_core.measurements.geometry import midpoint


def _angle_degrees(point_a, point_b) -> float:
    return math.degrees(
        math.atan2(point_b[1] - point_a[1], point_b[0] - point_a[0])
    )


def assess_image_quality(image_path: str, points: dict, *, source_type: str | None = None) -> dict:
    """Оценить геометрию головы, резкость, свет и размер лица."""
    with Image.open(Path(image_path)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    left_eye = midpoint(points["left_eye_outer"], points["left_eye_inner"])
    right_eye = midpoint(points["right_eye_inner"], points["right_eye_outer"])
    roll_degrees = _angle_degrees(left_eye, right_eye)

    face_left_x = points["face_left"][0]
    face_right_x = points["face_right"][0]
    face_width = abs(face_right_x - face_left_x)
    face_center_x = (face_left_x + face_right_x) / 2
    yaw_offset_ratio = (
        abs(points["nose_tip"][0] - face_center_x) / face_width
        if face_width
        else 1.0
    )

    point_values = list(points.values())
    min_x = max(0, round(min(point[0] for point in point_values)))
    max_x = min(image.width, round(max(point[0] for point in point_values)))
    min_y = max(0, round(min(point[1] for point in point_values)))
    max_y = min(image.height, round(max(point[1] for point in point_values)))
    padding_x = round((max_x - min_x) * 0.15)
    padding_y = round((max_y - min_y) * 0.15)
    left = max(0, min_x - padding_x)
    right = min(image.width, max_x + padding_x)
    top = max(0, min_y - padding_y)
    bottom = min(image.height, max_y + padding_y)
    face_gray = gray[top:bottom, left:right]
    if face_gray.size == 0:
        face_gray = gray

    blur_score = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
    brightness = float(face_gray.mean())
    contrast = float(face_gray.std())

    face_height = abs(points["chin"][1] - points["face_top"][1])
    face_coverage = (
        face_width * face_height / (image.width * image.height)
        if image.width and image.height
        else 0.0
    )
    mouth_width = abs(points["mouth_right"][0] - points["mouth_left"][0])
    mouth_opening = abs(points["lower_lip"][1] - points["upper_lip"][1])
    mouth_opening_ratio = mouth_opening / mouth_width if mouth_width else 1.0

    thresholds = QUALITY_THRESHOLDS
    checks = {
        "head_roll": abs(roll_degrees) <= thresholds["max_roll_degrees"],
        "head_yaw": yaw_offset_ratio <= thresholds["max_yaw_offset_ratio"],
        "sharpness": blur_score >= thresholds["min_blur_score"],
        "brightness": (
            thresholds["min_brightness"]
            <= brightness
            <= thresholds["max_brightness"]
        ),
        "contrast": contrast >= thresholds["min_contrast"],
        "face_size": face_coverage >= thresholds["min_face_coverage"],
        "neutral_expression": (
            mouth_opening_ratio <= thresholds["max_mouth_opening_ratio"]
        ),
        "resolution": min(image.size) >= thresholds["min_source_dimension"],
    }
    resolution_code = (
        "source_video_resolution_low"
        if source_type == "video_frame"
        else "source_image_resolution_low"
    )
    issue_codes = {
        "head_roll": "head_roll",
        "head_yaw": "head_yaw",
        "sharpness": "frame_blur",
        "brightness": "brightness",
        "contrast": "contrast",
        "face_size": "face_effective_resolution_low",
        "neutral_expression": "neutral_expression",
        "resolution": resolution_code,
    }
    labels = {
        "head_roll": "сильный наклон головы",
        "head_yaw": "лицо заметно повернуто",
        "sharpness": "frame_blur",
        "brightness": "неподходящая яркость",
        "contrast": "низкий контраст",
        "face_size": "face_effective_resolution_low",
        "neutral_expression": "рот заметно открыт; требуется нейтральное выражение",
        "resolution": resolution_code,
    }
    labels["resolution"] = (
        "\u043d\u0438\u0437\u043a\u043e\u0435 \u0438\u0441\u0445\u043e\u0434\u043d\u043e\u0435 \u0440\u0430\u0437\u0440\u0435\u0448\u0435\u043d\u0438\u0435 \u0432\u0438\u0434\u0435\u043e"
        if source_type == "video_frame"
        else "\u043d\u0438\u0437\u043a\u043e\u0435 \u0438\u0441\u0445\u043e\u0434\u043d\u043e\u0435 \u0440\u0430\u0437\u0440\u0435\u0448\u0435\u043d\u0438\u0435 \u0444\u043e\u0442\u043e\u0433\u0440\u0430\u0444\u0438\u0438"
    )
    issues = [labels[name] for name, passed in checks.items() if not passed]
    codes = [issue_codes[name] for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not issues else "warning",
        "issues": issues,
        "issue_codes": codes,
        "checks": checks,
        "metrics": {
            "roll_degrees": roll_degrees,
            "yaw_offset_ratio": yaw_offset_ratio,
            "blur_score": blur_score,
            "brightness": brightness,
            "contrast": contrast,
            "face_coverage": face_coverage,
            "face_width_px": face_width,
            "face_height_px": face_height,
            "face_area_ratio": face_coverage,
            "mouth_opening_ratio": mouth_opening_ratio,
            "source_width": image.width,
            "source_height": image.height,
        },
    }
