"""Адаптер MediaPipe Face Landmarker для одиночных фотографий."""

from pathlib import Path

from portrait_core.mesh import build_mesh, project_semantic_points
from portrait_core.topologies import MEDIAPIPE_478_CONTOURS

from .base import FacePointAdapter


class FaceAdapterError(RuntimeError):
    """Базовая ошибка извлечения точек лица."""


class FaceNotFoundError(FaceAdapterError):
    """На изображении не найдено лицо."""


class MultipleFacesError(FaceAdapterError):
    """На изображении обнаружено более одного лица."""


class ImageQualityError(FaceAdapterError):
    """Изображение не соответствует минимальным требованиям."""


class _BorrowedLandmarker:
    def __init__(self, landmarker):
        self.landmarker = landmarker

    def __enter__(self):
        return self.landmarker

    def __exit__(self, *_exc):
        return False


class MediaPipeAdapter(FacePointAdapter):
    """Преобразовать сетку MediaPipe в контракт измерительного ядра."""

    LANDMARK_INDEXES = {
        "face_left": 234,
        "face_right": 454,
        "face_top": 10,
        "chin": 152,
        "left_eye_outer": 33,
        "left_eye_inner": 133,
        "right_eye_inner": 362,
        "right_eye_outer": 263,
        "nose_tip": 1,
        "nose_bridge": 168,
        "nose_left": 98,
        "nose_right": 327,
        "mouth_left": 61,
        "mouth_right": 291,
        "upper_lip": 13,
        "lower_lip": 14,
        "jaw_left": 172,
        "jaw_right": 397,
        "left_brow_outer": 70,
        "left_brow_inner": 107,
        "right_brow_inner": 336,
        "right_brow_outer": 300,
    }

    def __init__(
        self,
        model_path: str,
        min_detection_confidence: float = 0.35,
        min_presence_confidence: float = 0.5,
        min_image_size: int = 256,
    ):
        self.model_path = Path(model_path)
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_image_size = min_image_size
        self._landmarker = None
        self._mp = None

    def prepare(self) -> None:
        if self._landmarker is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Модель MediaPipe не найдена: {self.model_path}")
        mp = self._load_mediapipe()
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=self.model_path.read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_confidence=self.min_presence_confidence,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._mp = mp

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            self._mp = None

    @staticmethod
    def _load_mediapipe():
        try:
            import mediapipe as mp
        except ImportError as error:
            raise FaceAdapterError(
                "MediaPipe не установлен. Выполните: pip install -r requirements.txt"
            ) from error
        return mp

    @classmethod
    def convert_mesh(
        cls,
        landmarks,
        width: int,
        height: int,
        offset_x: int = 0,
        offset_y: int = 0,
        coordinate_scale: float = 1.0,
    ) -> dict:
        """Преобразовать всю сетку детектора в Portrait Mesh Schema."""
        if width <= 0 or height <= 0:
            raise ValueError("Размер изображения должен быть положительным")
        if coordinate_scale <= 0:
            raise ValueError("Масштаб координат должен быть положительным")

        if not landmarks:
            raise FaceAdapterError("Детектор не вернул вершины лица")

        vertices = []
        for landmark in landmarks:
            vertices.append(
                [
                    float((landmark.x * width + offset_x) / coordinate_scale),
                    float((landmark.y * height + offset_y) / coordinate_scale),
                    float(
                        getattr(landmark, "z", 0.0)
                        * width
                        / coordinate_scale
                    ),
                ]
            )

        missing_indexes = [
            index
            for index in cls.LANDMARK_INDEXES.values()
            if index >= len(vertices)
        ]
        if missing_indexes:
            raise FaceAdapterError(
                f"Детектор не вернул вершину с индексом {min(missing_indexes)}"
            )

        return build_mesh(
            vertices,
            cls.LANDMARK_INDEXES,
            source="mediapipe-face-landmarker",
            source_topology=f"mediapipe-{len(vertices)}",
            image_width=round(width / coordinate_scale),
            image_height=round(height / coordinate_scale),
            metadata={
                "coordinate_scale": coordinate_scale,
                "crop_offset": [offset_x, offset_y],
                "contours": MEDIAPIPE_478_CONTOURS,
            },
        )

    @classmethod
    def convert_landmarks(cls, *args, **kwargs) -> dict:
        """Совместимый метод: вернуть 22 именованные точки из полной сетки."""
        return project_semantic_points(cls.convert_mesh(*args, **kwargs))

    @staticmethod
    def _candidate_regions(width: int, height: int) -> list[tuple[int, int, int, int]]:
        """Вернуть полный кадр и перекрывающиеся области для мелких лиц."""
        regions = [(0, 0, width, height)]
        crop_width = round(width * 0.68)
        crop_height = round(height * 0.78)
        x_positions = (0, (width - crop_width) // 2, width - crop_width)
        y_positions = (0, (height - crop_height) // 2, height - crop_height)

        for y in y_positions:
            for x in x_positions:
                region = (x, y, x + crop_width, y + crop_height)
                if region not in regions:
                    regions.append(region)
        return regions

    @staticmethod
    def _to_mp_image(mp, image):
        """Создать MediaPipe Image без передачи Unicode-пути нативному коду."""
        import numpy as np

        rgb = np.ascontiguousarray(image.convert("RGB"))
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def extract_mesh(self, image_path: str) -> dict:
        """Найти одно лицо и вернуть полную сетку в проектном формате."""
        image_file = Path(image_path)
        if not image_file.is_file():
            raise FileNotFoundError(f"Фотография не найдена: {image_file}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Модель MediaPipe не найдена: {self.model_path}")

        from PIL import Image, ImageOps

        self.prepare()
        mp = self._mp
        with Image.open(image_file) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        if min(image.size) < 64:
            raise ImageQualityError(
                "Короткая сторона фотографии должна быть не меньше 64 пикселей"
            )
        coordinate_scale = 1.0
        if min(image.size) < self.min_image_size:
            coordinate_scale = self.min_image_size / min(image.size)
            image = image.resize(
                (
                    round(image.width * coordinate_scale),
                    round(image.height * coordinate_scale),
                ),
                Image.Resampling.LANCZOS,
            )

        with _BorrowedLandmarker(self._landmarker) as landmarker:
            for left, top, right, bottom in self._candidate_regions(*image.size):
                crop = image.crop((left, top, right, bottom))
                result = landmarker.detect(self._to_mp_image(mp, crop))
                face_count = len(result.face_landmarks)
                if face_count > 1:
                    raise MultipleFacesError(
                        "На фотографии найдено несколько лиц; требуется одно лицо"
                    )
                if face_count == 1:
                    mesh = self.convert_mesh(
                        result.face_landmarks[0],
                        width=crop.width,
                        height=crop.height,
                        offset_x=left,
                        offset_y=top,
                        coordinate_scale=coordinate_scale,
                    )
                    mesh["image_size"] = {
                        "width": round(image.width / coordinate_scale),
                        "height": round(image.height / coordinate_scale),
                    }
                    return mesh

        raise FaceNotFoundError(
            "На фотографии не найдено лицо. Попробуйте кадр крупнее, анфас "
            "и при более равномерном освещении."
        )
