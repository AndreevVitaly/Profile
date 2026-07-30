"""GUI Dataset Builder для платформы Profile."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Поддерживаем запуск как модуль и напрямую через Run/абсолютный путь.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from apps.dataset_builder.builder import StopRequested, build_dataset, is_url


class BuildWorker(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        input_path: str,
        output_dir: str,
        backend: str,
        model_path: str | None,
        topology_path: str | None,
        frame_step: int,
        copy_images: bool,
        dominant_face_track: bool,
        min_track_length: int,
        min_video_height: int,
        allow_quality_fallback: bool,
        frame_selection_mode: str,
        selection_profile: str,
        target_selected_frames: int,
        min_temporal_distance_seconds: float,
        max_abs_yaw_deg: float,
        max_abs_pitch_deg: float,
        max_abs_roll_deg: float,
        require_closed_mouth: bool,
        require_open_eyes: bool,
        use_gaze_score: bool,
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.backend = backend
        self.model_path = model_path
        self.topology_path = topology_path
        self.frame_step = frame_step
        self.copy_images = copy_images
        self.dominant_face_track = dominant_face_track
        self.min_track_length = min_track_length
        self.min_video_height = min_video_height
        self.allow_quality_fallback = allow_quality_fallback
        self.frame_selection_mode = frame_selection_mode
        self.selection_profile = selection_profile
        self.target_selected_frames = target_selected_frames
        self.min_temporal_distance_seconds = min_temporal_distance_seconds
        self.max_abs_yaw_deg = max_abs_yaw_deg
        self.max_abs_pitch_deg = max_abs_pitch_deg
        self.max_abs_roll_deg = max_abs_roll_deg
        self.require_closed_mouth = require_closed_mouth
        self.require_open_eyes = require_open_eyes
        self.use_gaze_score = use_gaze_score
        self._stop = False

    @pyqtSlot()
    def run(self) -> None:
        try:
            summary = build_dataset(
                self.input_path,
                self.output_dir,
                backend=self.backend,
                model_path=self.model_path,
                topology_path=self.topology_path,
                frame_step=self.frame_step,
                copy_images=self.copy_images,
                dominant_face_track=self.dominant_face_track,
                min_track_length=self.min_track_length,
                video_quality="best",
                min_video_height=self.min_video_height,
                allow_quality_fallback=self.allow_quality_fallback,
                frame_selection_mode=self.frame_selection_mode,
                selection_profile=self.selection_profile,
                target_selected_frames=self.target_selected_frames,
                min_temporal_distance_seconds=self.min_temporal_distance_seconds,
                max_abs_yaw_deg=self.max_abs_yaw_deg,
                max_abs_pitch_deg=self.max_abs_pitch_deg,
                max_abs_roll_deg=self.max_abs_roll_deg,
                require_closed_mouth=self.require_closed_mouth,
                require_open_eyes=self.require_open_eyes,
                use_gaze_score=self.use_gaze_score,
                log=self.log.emit,
                progress=self._emit_progress,
                should_stop=lambda: self._stop,
            )
            self.finished.emit(summary)
        except StopRequested as error:
            self.failed.emit(str(error))
        except Exception as error:  # noqa: BLE001 - ошибка должна попасть в GUI.
            self.failed.emit(str(error))

    def stop(self) -> None:
        self._stop = True

    def _emit_progress(self, current: int, total: int) -> None:
        self.progress.emit(int(current / max(total, 1) * 100))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Profile Dataset Builder")
        self.resize(920, 680)
        self.thread: QThread | None = None
        self.worker: BuildWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        source_box = QGroupBox("Источник")
        source_layout = QGridLayout(source_box)
        self.file_radio = QRadioButton("Файл / видео / URL")
        self.folder_radio = QRadioButton("Папка изображений")
        self.folder_radio.setChecked(True)
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Папка изображений, файл изображения, видео или URL")
        browse_file = QPushButton("Файл...")
        browse_folder = QPushButton("Папка...")
        browse_file.clicked.connect(self._choose_file)
        browse_folder.clicked.connect(self._choose_folder)
        source_layout.addWidget(self.folder_radio, 0, 0)
        source_layout.addWidget(self.file_radio, 0, 1)
        source_layout.addWidget(self.input_path, 1, 0, 1, 2)
        source_layout.addWidget(browse_file, 1, 2)
        source_layout.addWidget(browse_folder, 1, 3)
        layout.addWidget(source_box)

        output_box = QGroupBox("Результат")
        output_layout = QGridLayout(output_box)
        self.output_path = QLineEdit("dataset")
        output_browse = QPushButton("Выбрать...")
        output_browse.clicked.connect(self._choose_output)
        output_layout.addWidget(QLabel("Папка результата"), 0, 0)
        output_layout.addWidget(self.output_path, 0, 1)
        output_layout.addWidget(output_browse, 0, 2)
        layout.addWidget(output_box)

        settings_box = QGroupBox("Настройки")
        settings_layout = QFormLayout(settings_box)
        self.frame_step = QSpinBox()
        self.frame_step.setRange(1, 100000)
        self.frame_step.setValue(24)
        self.copy_images = QCheckBox("копировать изображения в passed / warning")
        self.copy_images.setChecked(True)
        self.dominant_face_track = QCheckBox("для видео анализировать часто повторяющееся лицо")
        self.min_track_length = QSpinBox()
        self.min_track_length.setRange(1, 100000)
        self.min_track_length.setValue(3)
        self.min_video_height = QSpinBox()
        self.min_video_height.setRange(144, 4320)
        self.min_video_height.setValue(720)
        self.allow_quality_fallback = QCheckBox("если высокого качества нет, брать лучшее доступное")
        self.allow_quality_fallback.setChecked(True)
        self.frame_selection_mode = QComboBox()
        self.frame_selection_mode.addItems(["fixed_step", "quality_profile"])
        self.selection_profile = QComboBox()
        self.selection_profile.addItems(["frontal_neutral"])
        self.target_selected_frames = QSpinBox()
        self.target_selected_frames.setRange(1, 100000)
        self.target_selected_frames.setValue(100)
        self.min_temporal_distance = QSpinBox()
        self.min_temporal_distance.setRange(0, 60)
        self.min_temporal_distance.setValue(1)
        self.max_abs_yaw = QSpinBox()
        self.max_abs_yaw.setRange(0, 90)
        self.max_abs_yaw.setValue(15)
        self.max_abs_pitch = QSpinBox()
        self.max_abs_pitch.setRange(0, 90)
        self.max_abs_pitch.setValue(12)
        self.max_abs_roll = QSpinBox()
        self.max_abs_roll.setRange(0, 90)
        self.max_abs_roll.setValue(10)
        self.require_closed_mouth = QCheckBox("исключать открытый рот")
        self.require_closed_mouth.setChecked(True)
        self.require_open_eyes = QCheckBox("требовать открытые глаза")
        self.require_open_eyes.setChecked(True)
        self.use_gaze_score = QCheckBox("использовать оценку взгляда")
        self.use_gaze_score.setChecked(True)
        self.backend_mediapipe = QRadioButton("MediaPipe")
        self.backend_onnx = QRadioButton("ONNX")
        self.backend_mediapipe.setChecked(True)
        backend_row = QHBoxLayout()
        backend_row.addWidget(self.backend_mediapipe)
        backend_row.addWidget(self.backend_onnx)
        backend_row.addStretch(1)
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Необязательно: путь к модели")
        model_browse = QPushButton("Модель...")
        model_browse.clicked.connect(self._choose_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path)
        model_row.addWidget(model_browse)
        self.topology_path = QLineEdit()
        self.topology_path.setPlaceholderText("Для ONNX: путь к topology/sidecar JSON")
        topology_browse = QPushButton("Topology...")
        topology_browse.clicked.connect(self._choose_topology)
        topology_row = QHBoxLayout()
        topology_row.addWidget(self.topology_path)
        topology_row.addWidget(topology_browse)
        settings_layout.addRow("Backend", backend_row)
        settings_layout.addRow("Шаг кадров для видео", self.frame_step)
        settings_layout.addRow("", self.copy_images)
        settings_layout.addRow("", self.dominant_face_track)
        settings_layout.addRow("Мин. повторов лица", self.min_track_length)
        settings_layout.addRow("Мин. высота видео", self.min_video_height)
        settings_layout.addRow("", self.allow_quality_fallback)
        settings_layout.addRow("Режим выбора кадров", self.frame_selection_mode)
        settings_layout.addRow("Профиль", self.selection_profile)
        settings_layout.addRow("Целевое число кадров", self.target_selected_frames)
        settings_layout.addRow("Мин. дистанция, сек", self.min_temporal_distance)
        settings_layout.addRow("Порог yaw", self.max_abs_yaw)
        settings_layout.addRow("Порог pitch", self.max_abs_pitch)
        settings_layout.addRow("Порог roll", self.max_abs_roll)
        settings_layout.addRow("", self.require_closed_mouth)
        settings_layout.addRow("", self.require_open_eyes)
        settings_layout.addRow("", self.use_gaze_score)
        settings_layout.addRow("Модель", model_row)
        settings_layout.addRow("Topology", topology_row)
        layout.addWidget(settings_box)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("START")
        self.stop_button = QPushButton("STOP")
        self.open_button = QPushButton("Открыть папку результата")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.open_button.clicked.connect(self._open_result)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.open_button)
        layout.addLayout(buttons)

        counters = QHBoxLayout()
        self.total_label = QLabel("Всего: 0")
        self.passed_label = QLabel("Passed: 0")
        self.warning_label = QLabel("Warning: 0")
        self.rejected_label = QLabel("Rejected: 0")
        counters.addWidget(self.total_label)
        counters.addWidget(self.passed_label)
        counters.addWidget(self.warning_label)
        counters.addWidget(self.rejected_label)
        counters.addStretch(1)
        layout.addLayout(counters)

        self.progress = QProgressBar()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.progress)
        layout.addWidget(self.log, 1)

        self.setCentralWidget(root)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать файл",
            "",
            "Media (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.webm);;All files (*.*)",
        )
        if path:
            self.input_path.setText(path)
            self.file_radio.setChecked(True)

    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Выбрать папку изображений")
        if path:
            self.input_path.setText(path)
            self.folder_radio.setChecked(True)

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Выбрать папку результата")
        if path:
            self.output_path.setText(path)

    def _choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать модель",
            "",
            "Model (*.task *.onnx);;All files (*.*)",
        )
        if path:
            self.model_path.setText(path)

    def _choose_topology(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать topology JSON",
            "",
            "JSON (*.json);;All files (*.*)",
        )
        if path:
            self.topology_path.setText(path)

    def _start(self) -> None:
        input_path = self.input_path.text().strip()
        output_dir = self.output_path.text().strip()
        if not input_path:
            QMessageBox.warning(self, "Dataset Builder", "Источник не указан.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Dataset Builder", "Папка результата не указана.")
            return
        if not is_url(input_path) and not Path(input_path).exists():
            QMessageBox.warning(self, "Dataset Builder", "Источник не найден.")
            return

        self.progress.setRange(0, 0)
        self.log.clear()
        self._update_counters(None)
        self.thread = QThread(self)
        self.worker = BuildWorker(
            input_path=input_path,
            output_dir=output_dir,
            backend="onnx" if self.backend_onnx.isChecked() else "mediapipe",
            model_path=self.model_path.text().strip() or None,
            topology_path=self.topology_path.text().strip() or None,
            frame_step=self.frame_step.value(),
            copy_images=self.copy_images.isChecked(),
            dominant_face_track=self.dominant_face_track.isChecked(),
            min_track_length=self.min_track_length.value(),
            min_video_height=self.min_video_height.value(),
            allow_quality_fallback=self.allow_quality_fallback.isChecked(),
            frame_selection_mode=self.frame_selection_mode.currentText(),
            selection_profile=self.selection_profile.currentText(),
            target_selected_frames=self.target_selected_frames.value(),
            min_temporal_distance_seconds=float(self.min_temporal_distance.value()),
            max_abs_yaw_deg=float(self.max_abs_yaw.value()),
            max_abs_pitch_deg=float(self.max_abs_pitch.value()),
            max_abs_roll_deg=float(self.max_abs_roll.value()),
            require_closed_mouth=self.require_closed_mouth.isChecked(),
            require_open_eyes=self.require_open_eyes.isChecked(),
            use_gaze_score=self.use_gaze_score.isChecked(),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self._set_progress)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.thread.start()

    def _stop(self) -> None:
        if self.worker:
            self.worker.stop()
            self._append_log("Остановка запрошена...")

    def _finished(self, summary: dict) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._update_counters(summary)
        self._append_log(f"Готово: {summary.get('output_dir')}")

    def _failed(self, message: str) -> None:
        self.progress.setRange(0, 100)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._append_log(message)
        QMessageBox.warning(self, "Dataset Builder", message)

    def _set_progress(self, value: int) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(value)

    def _append_log(self, message: str) -> None:
        self.log.append(message)

    def _update_counters(self, summary: dict | None) -> None:
        if not summary:
            self.total_label.setText("Всего: 0")
            self.passed_label.setText("Passed: 0")
            self.warning_label.setText("Warning: 0")
            self.rejected_label.setText("Rejected: 0")
            return
        statuses = summary.get("statuses", {})
        self.total_label.setText(f"Всего: {summary.get('total_images', 0)}")
        self.passed_label.setText(f"Passed: {statuses.get('passed', 0)}")
        self.warning_label.setText(f"Warning: {statuses.get('warning', 0)}")
        self.rejected_label.setText(f"Rejected: {statuses.get('rejected', 0)}")

    def _open_result(self) -> None:
        path = Path(self.output_path.text().strip() or "dataset").resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)



def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
