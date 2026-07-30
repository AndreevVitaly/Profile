"""Video preprocessing for dominant non-identifying face tracks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from portrait_core.tracking.selector import FaceObservation, select_dominant_track


def select_dominant_face_track(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    frame_step: int = 24,
    min_track_length: int = 3,
    crop_padding: float = 0.35,
    max_scan_width: int = 960,
    log: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[Path]:
    """Detect repeated face boxes and save crops from the dominant track.

    The output is a technical observation series. It is not identity
    recognition and it does not compare faces with any external database.
    """
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-contrib-python is required for video face-track selection") from error

    video = Path(video_path)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    observations: list[FaceObservation] = []
    frame_index = 0
    scanned = 0
    frame_step = max(1, frame_step)
    max_scan_width = max(320, max_scan_width)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    try:
        while True:
            if should_stop and should_stop():
                break
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                scanned += 1
                height, width = frame.shape[:2]
                scan_frame, scale = _resize_for_scan(frame, max_scan_width)
                gray = cv2.cvtColor(scan_frame, cv2.COLOR_BGR2GRAY)
                detections = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40),
                )
                for x, y, box_width, box_height in detections:
                    observations.append(
                        FaceObservation(
                            frame_index=frame_index,
                            bbox=(
                                float(x) / scale,
                                float(y) / scale,
                                float(box_width) / scale,
                                float(box_height) / scale,
                            ),
                            frame_size=(width, height),
                        )
                    )
                if log and scanned % 50 == 0:
                    suffix = f"/{total_frames}" if total_frames else ""
                    log(
                        "face-track scan: "
                        f"processed {scanned} sampled frames, "
                        f"source frame {frame_index}{suffix}, "
                        f"detections {len(observations)}"
                    )
            frame_index += 1
    finally:
        capture.release()

    track = select_dominant_track(
        observations,
        min_count=min_track_length,
        max_frame_gap=max(3, frame_step * 3),
    )
    if track is None:
        if log:
            log("dominant face-track was not found")
        _write_manifest(target, video, None, observations, [])
        return []

    selected_by_frame = {item.frame_index: item for item in track.observations}
    output_paths = _save_selected_track_crops(
        video,
        target,
        selected_by_frame,
        track.track_id,
        crop_padding,
        log=log,
        should_stop=should_stop,
    )
    _write_manifest(target, video, track, observations, output_paths)
    return output_paths


def _resize_for_scan(frame, max_width: int):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0
    try:
        import cv2
    except ImportError:
        return frame, 1.0
    scale = max_width / float(width)
    resized = cv2.resize(
        frame,
        (max_width, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _save_selected_track_crops(
    video: Path,
    target: Path,
    selected_by_frame: dict[int, FaceObservation],
    track_id: str,
    crop_padding: float,
    *,
    log: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[Path]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-contrib-python is required for video face-track selection") from error

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video: {video}")

    output_paths: list[Path] = []
    wanted = set(selected_by_frame)
    frame_index = 0
    try:
        while wanted:
            if should_stop and should_stop():
                break
            ok, frame = capture.read()
            if not ok:
                break
            observation = selected_by_frame.get(frame_index)
            if observation is not None:
                crop = _crop_with_padding(frame, observation.bbox, crop_padding)
                output_path = target / f"{len(output_paths) + 1:04d}_track_{track_id}_frame{frame_index:06d}.jpg"
                _write_jpeg(output_path, crop)
                output_paths.append(output_path)
                wanted.remove(frame_index)
                if log:
                    log(f"dominant face-track crop saved: {output_path.name}")
            frame_index += 1
    finally:
        capture.release()
    return output_paths

def _write_jpeg(output_path: Path, image) -> None:
    """Write JPEG through Python I/O so Windows handles Unicode paths reliably."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-contrib-python is required for video face-track selection") from error

    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"Could not encode face-track crop: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded.tobytes())
    if not output_path.is_file():
        raise RuntimeError(f"Could not write face-track crop: {output_path}")

def _crop_with_padding(frame, bbox: tuple[float, float, float, float], padding: float):
    x, y, width, height = bbox
    frame_height, frame_width = frame.shape[:2]
    pad_x = width * padding
    pad_y = height * padding
    left = max(0, int(round(x - pad_x)))
    top = max(0, int(round(y - pad_y)))
    right = min(frame_width, int(round(x + width + pad_x)))
    bottom = min(frame_height, int(round(y + height + pad_y)))
    return frame[top:bottom, left:right]


def _write_manifest(
    output_dir: Path,
    video_path: Path,
    track,
    observations: list[FaceObservation],
    output_paths: list[Path],
) -> None:
    payload = {
        "schema": "profile.face_track_selection.v1",
        "source_video": str(video_path),
        "policy": "geometry-only dominant track selection; no identity recognition",
        "detections": len(observations),
        "selected_track": None
        if track is None
        else {
            "track_id": track.track_id,
            "count": track.count,
            "score": round(track.score, 6),
            "mean_area": round(track.mean_area, 6),
            "mean_centrality": round(track.mean_centrality, 6),
            "continuity": round(track.continuity, 6),
        },
        "frames": [path.name for path in output_paths],
    }
    (output_dir / "face_track_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
