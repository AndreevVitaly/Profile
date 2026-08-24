"""Human-readable illustrated PDF report for an ORION Dataset Archive."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QImage, QPageLayout, QPageSize, QPainter, QPen, QPdfWriter

from portrait_core.report_pack import build_report_pack


MEASUREMENT_LINES = (
    ("Ширина лица", "face_left", "face_right", "#00a6a6"),
    ("Высота лица", "face_top", "chin", "#12a150"),
    ("IPD", "left_eye_inner", "right_eye_inner", "#2f6bff"),
    ("Внешняя ширина глаз", "left_eye_outer", "right_eye_outer", "#557cff"),
    ("Ширина носа", "nose_left", "nose_right", "#e09f00"),
    ("Длина носа", "nose_bridge", "nose_tip", "#ffb52e"),
    ("Ширина рта", "mouth_left", "mouth_right", "#e5484d"),
    ("Ширина челюсти", "jaw_left", "jaw_right", "#12a150"),
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _repair_text(value: Any) -> str:
    text = str(value)
    if "Р" not in text and "С" not in text:
        return text
    try:
        return text.encode("cp1251").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _flatten_numbers(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_numbers(child, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        rows.append((prefix, float(value)))
    return rows


def _format_value(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.3f}"
    return f"{value:.5f}"


class _PdfCanvas:
    def __init__(self, output: Path):
        self.writer = QPdfWriter(str(output))
        self.writer.setResolution(150)
        self.writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        self.writer.setPageOrientation(QPageLayout.Orientation.Landscape)
        self.painter = QPainter(self.writer)
        if not self.painter.isActive():
            raise RuntimeError(f"Не удалось создать PDF: {output}")
        self.width = self.writer.width()
        self.height = self.writer.height()
        self.margin = 70
        self.first_page = True

    def page(self, title: str, subtitle: str | None = None) -> float:
        if not self.first_page:
            self.writer.newPage()
        self.first_page = False
        self.painter.fillRect(QRectF(0, 0, self.width, self.height), QColor("#ffffff"))
        self.painter.setPen(QColor("#152238"))
        self.painter.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.painter.drawText(QRectF(self.margin, 45, self.width - 2 * self.margin, 48), title)
        y = 100.0
        if subtitle:
            self.painter.setFont(QFont("Arial", 8))
            self.painter.setPen(QColor("#53657d"))
            self.painter.drawText(QRectF(self.margin, y, self.width - 2 * self.margin, 40), subtitle)
            y += 42
        self.painter.setPen(QPen(QColor("#d8e0ea"), 2))
        self.painter.drawLine(self.margin, int(y), self.width - self.margin, int(y))
        return y + 25

    def lines(self, title: str, lines: Iterable[str], *, subtitle: str | None = None) -> None:
        y = self.page(title, subtitle)
        self.painter.setFont(QFont("Arial", 9))
        line_height = 27
        for raw in lines:
            text = _repair_text(raw)
            if y + line_height > self.height - self.margin:
                y = self.page(title + " — продолжение")
                self.painter.setFont(QFont("Arial", 9))
            color = "#b42318" if text.startswith("✗") else "#8a5a00" if text.startswith("!") else "#152238"
            self.painter.setPen(QColor(color))
            self.painter.drawText(QRectF(self.margin, y, self.width - 2 * self.margin, line_height), text)
            y += line_height

    def close(self) -> None:
        self.painter.end()


def _summary_lines(dataset: dict, summary: dict, pack: dict) -> list[str]:
    statuses = summary.get("statuses") or {}
    total = int(summary.get("total_images") or len(dataset.get("items") or []))
    reports = int(summary.get("created_reports") or pack.get("reports_count") or 0)
    backend = dataset.get("analysis_backend") or {}
    model = backend.get("model") or {}
    lines = [
        f"Dataset: {dataset.get('id')} / UUID: {dataset.get('uuid')}",
        f"Источник: {dataset.get('source')}",
        f"Наблюдения: {total}; PFR: {reports}; полнота: {(reports / total * 100 if total else 0):.1f}%",
        f"Passed: {statuses.get('passed', 0)}; Warning: {statuses.get('warning', 0)}; Rejected: {statuses.get('rejected', 0)}",
        f"Backend: {backend.get('name')}; model: {backend.get('model_id')}; validation: {model.get('validation_status')}",
        f"SHA-256 модели: {model.get('sha256')}",
        "Назначение: исследование геометрии лица и устойчивости измерений; не является выводом о личности.",
    ]
    resolution = dataset.get("face_effective_resolution") or {}
    if resolution.get("samples"):
        lines.extend([
            f"Медианный размер лица: {resolution.get('median_face_width_px', 0):.1f} × {resolution.get('median_face_height_px', 0):.1f} px",
            f"Медианная доля лица в кадре: {resolution.get('median_face_area_ratio', 0) * 100:.1f}%",
        ])
    return lines


def _quality_lines(summary: dict) -> list[str]:
    rows = summary.get("rows") or []
    codes = Counter(code for row in rows for code in (row.get("issue_codes") or []))
    lines = ["Частота диагностических кодов:"]
    lines.extend(f"! {code}: {count}" for code, count in codes.most_common())
    lines.extend([
        "",
        "Интерпретация: предупреждения описывают пригодность кадра, а не свойства человека.",
        "Высокая доля warning требует раздельного анализа passed/warning и проверки чувствительности выводов.",
    ])
    return lines


def _hypothesis_lines(pack: dict) -> list[str]:
    lic = pack.get("lic_stability") or {}
    ranking = lic.get("ranking") or []
    by_name = {row.get("name"): row for row in ranking}
    best = lic.get("best_candidate")
    top = (pack.get("point_stability") or {}).get("top_10") or []
    top_names = [row.get("name") for row in top[:10]]
    ipd = by_name.get("ipd") or {}
    nose = next((row for row in top if row.get("name") in {"nose_bridge", "nose_tip"}), None)
    return [
        "H-0001 — IPD как база нормализации:",
        f"  Результат: лучший текущий кандидат = {best}; CV(IPD) = {ipd.get('coefficient_of_variation', 'n/a')}.",
        "  Вывод: предварительная поддержка, но не подтверждение; необходимы passed-only и межсубъектные серии.",
        "H-0002 — LIC как каркас из нескольких устойчивых точек:",
        f"  TOP устойчивых точек: {', '.join(str(name) for name in top_names[:5]) or 'нет данных'}.",
        "  Вывод: гипотеза проверяема на этом наборе; устойчивость графа ещё не рассчитана.",
        "H-0003 — устойчивость внутренних уголков глаз:",
        f"  Позиции в TOP-10: left_eye_inner={top_names.index('left_eye_inner') + 1 if 'left_eye_inner' in top_names else 'нет'}, right_eye_inner={top_names.index('right_eye_inner') + 1 if 'right_eye_inner' in top_names else 'нет'}.",
        "  Вывод: предварительно поддерживается текущей серией.",
        "H-0004 — переносица/нос как часть LIC:",
        f"  Результат: {nose if nose else 'точки носа не вошли в TOP-10'}.",
        "  Вывод: требуется отдельное сравнение nose_bridge, nose_tip, chin и mouth_center.",
        "H-0006 — деградация устойчивости при снижении качества:",
        "  Вывод: набор содержит passed/warning, но статистику необходимо рассчитывать раздельно по группам.",
        "H-0007 — Dataset Builder как научный прибор:",
        "  Вывод: provenance, PFR и диагностические статусы обеспечивают воспроизводимую основу эксперимента.",
    ]


def _proposal_lines(pack: dict) -> list[str]:
    return [
        "1. Повторить отбор в режиме quality_profile/frontal_neutral и сравнить его с fixed_step.",
        "2. Рассчитывать LIC и landmark stability отдельно для passed и warning.",
        "3. Выполнить дедупликацию соседних кадров до статистического анализа.",
        "4. Проверить IPD на нескольких людях и разных видео; одна серия не подтверждает универсальность базы.",
        "5. Сравнить CV IPD, outer_eye_distance, inner_eye_distance и face_width.",
        "6. Проверить устойчивость nose_bridge, nose_tip, chin и mouth_center.",
        "7. Не использовать геометрические показатели для психологических, HR или идентификационных выводов.",
    ]


def _draw_frame_page(canvas: _PdfCanvas, index: int, total: int, dataset_dir: Path, item: dict) -> None:
    status = item.get("status", "unknown")
    image_path = dataset_dir / str(item.get("image_path") or "")
    pfr_path = dataset_dir / str(item.get("pfr_path") or "") if item.get("pfr_path") else None
    report = _read(pfr_path) if pfr_path and pfr_path.is_file() else None
    y = canvas.page(f"Кадр {index}/{total}: {image_path.name}", f"Статус: {status}; frame={item.get('frame_index')}; PFR={item.get('pfr_id')}")
    painter = canvas.painter
    image_box = QRectF(canvas.margin, y, canvas.width * 0.53, canvas.height - y - canvas.margin)
    image = QImage(str(image_path))
    if not image.isNull():
        scaled = image.size().scaled(int(image_box.width()), int(image_box.height()), Qt.AspectRatioMode.KeepAspectRatio)
        target = QRectF(image_box.x() + (image_box.width() - scaled.width()) / 2,
                        image_box.y() + (image_box.height() - scaled.height()) / 2,
                        scaled.width(), scaled.height())
        painter.drawImage(target, image)
        if report:
            points = ((report.get("geometry") or {}).get("points") or report.get("points") or {})
            iw = float(image.width() or 1)
            ih = float(image.height() or 1)
            for label, start, end, color in MEASUREMENT_LINES:
                if start not in points or end not in points:
                    continue
                x1, y1 = points[start][:2]
                x2, y2 = points[end][:2]
                painter.setPen(QPen(QColor(color), 3))
                ax, ay = target.x() + x1 / iw * target.width(), target.y() + y1 / ih * target.height()
                bx, by = target.x() + x2 / iw * target.width(), target.y() + y2 / ih * target.height()
                painter.drawLine(int(ax), int(ay), int(bx), int(by))
                painter.setFont(QFont("Arial", 6, QFont.Weight.Bold))
                painter.drawText(QRectF((ax + bx) / 2 + 4, (ay + by) / 2 - 10, 180, 20), label)
                painter.setBrush(QColor(color))
                painter.drawEllipse(QRectF(ax - 4, ay - 4, 8, 8)); painter.drawEllipse(QRectF(bx - 4, by - 4, 8, 8))
    else:
        painter.setPen(QColor("#b42318")); painter.drawText(image_box, "Изображение недоступно")

    x = canvas.width * 0.58
    right_width = canvas.width - x - canvas.margin
    painter.setFont(QFont("Arial", 7))
    ty = y
    lines = [f"Проблемы: {', '.join(_repair_text(v) for v in item.get('issues') or []) or 'нет'}"]
    if report:
        quality = report.get("quality") or {}
        lines.append("Качество:")
        lines.extend(f"  {name}: {_format_value(value)}" for name, value in _flatten_numbers(quality.get("metrics") or {}))
        lines.append("Все измерения:")
        lines.extend(f"  {name}: {_format_value(value)}" for name, value in _flatten_numbers(report.get("measurements") or {}))
        lic = report.get("lic_core") or {}
        lines.append(f"LIC recommended_base: {lic.get('recommended_base')}")
    else:
        lines.append("PFR не создан; измерения отсутствуют.")
    for line in lines:
        if ty > canvas.height - canvas.margin - 18:
            break
        painter.setPen(QColor("#152238"))
        painter.drawText(QRectF(x, ty, right_width, 20), line)
        ty += 18


def _draw_measurement_table(canvas: _PdfCanvas, index: int, total: int, image_name: str, report: dict) -> None:
    rows = _flatten_numbers(report.get("measurements") or {})
    if not rows:
        return
    y = canvas.page(f"Все измерения кадра {index}/{total}", image_name)
    painter = canvas.painter
    painter.setFont(QFont("Arial", 6))
    available = canvas.height - y - canvas.margin
    line_height = 19
    per_column = max(1, int(available // line_height))
    columns = max(1, math.ceil(len(rows) / per_column))
    column_width = (canvas.width - 2 * canvas.margin) / columns
    for position, (name, value) in enumerate(rows):
        column = position // per_column
        row = position % per_column
        x = canvas.margin + column * column_width
        ry = y + row * line_height
        painter.setPen(QColor("#152238"))
        painter.drawText(QRectF(x, ry, column_width - 12, line_height), f"{name}: {_format_value(value)}")


def build_dataset_pdf_report(dataset_directory: str | Path, output_path: str | Path | None = None) -> Path:
    application = QGuiApplication.instance() or QGuiApplication([])
    dataset_dir = Path(dataset_directory).resolve()
    dataset = _read(dataset_dir / "dataset.json")
    summary = _read(dataset_dir / "summary.json")
    pack = build_report_pack(str(dataset_dir), include_frames=False)
    output = Path(output_path) if output_path else dataset_dir / "detailed_report.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = _PdfCanvas(output)
    try:
        canvas.lines("ORION — подробный отчёт Dataset", _summary_lines(dataset, summary, pack), subtitle="Автоматический исследовательский отчёт")
        canvas.lines("Качество наблюдений", _quality_lines(summary))
        canvas.lines("Работа с гипотезами", _hypothesis_lines(pack))
        canvas.lines("Выводы и предложения", _proposal_lines(pack))
        items = dataset.get("items") or []
        for index, item in enumerate(items, 1):
            _draw_frame_page(canvas, index, len(items), dataset_dir, item)
            if item.get("pfr_path"):
                report_path = dataset_dir / str(item["pfr_path"])
                if report_path.is_file():
                    _draw_measurement_table(canvas, index, len(items), Path(str(item.get("image_path") or "")).name, _read(report_path))
    finally:
        canvas.close()
    try:
        relative = output.resolve().relative_to(dataset_dir).as_posix()
    except ValueError:
        relative = None
    if relative is not None:
        summary["pdf_report"] = relative
        artifacts = [item for item in (dataset.get("artifacts") or []) if item.get("type") != "detailed_pdf_report"]
        artifacts.append({"type": "detailed_pdf_report", "path": relative})
        dataset["artifacts"] = artifacts
        (dataset_dir / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
        (dataset_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build illustrated ORION Dataset PDF report")
    parser.add_argument("dataset_directory")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(build_dataset_pdf_report(args.dataset_directory, args.output))


if __name__ == "__main__":
    main()
