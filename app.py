import html
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from PyQt5.QtCore import (
    QEvent,
    QObject,
    QPropertyAnimation,
    QThread,
    QTimer,
    QUrl,
    Qt,
    pyqtProperty,
    pyqtSignal,
    QSize,
)
from PyQt5.QtGui import QDesktopServices, QIntValidator, QKeySequence, QPainter, QPen, QTextCursor, QColor, QFont
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QToolButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QFrame,
)

from chronam import download_data
from chronam.config import DEFAULT_CSV_FILENAME, default_csv_path
from chronam.map_create import create_map, _build_time_index, _parse_date
from chronam.collocate import run_collocation, build_collocation_output_paths
from chronam.topics import (
    TopicModelParameters,
    build_topic_model_output_paths,
    run_topic_model,
)
from chronam.exceptions import OperationCancelledError
from chronam.utils import term_directory_name
from chronam.metrics import metric_total_for_dates


def reveal_in_file_manager(path: str):
    if not path or not os.path.exists(path):
        return
    try:
        if sys.platform == 'darwin':
            subprocess.run(['open', '-R', path], check=False)
        elif os.name == 'nt':
            norm = os.path.normpath(path)
            subprocess.run(['explorer', f'/select,{norm}'], check=False)
        else:
            directory = os.path.dirname(path) or '.'
            subprocess.run(['xdg-open', directory], check=False)
    except Exception:
        pass

DATASET_FOLDER_WARNING = (
    "Select the folder containing the AmericanStories parquet files. "
    "If you have already set a folder location, the software does not recognize the parquet files. "
    "Ensure the folder is unzipped and accessible."
)


def resolve_locations_csv(parent: Optional[QWidget]) -> Optional[str]:
    candidates = []
    if parent is not None:
        explicit = getattr(parent, 'locations_csv_path', None)
        if explicit:
            candidates.append(explicit)
    # Always prefer packaged default next
    candidates.append(default_csv_path())
    if parent is not None:
        dataset = getattr(parent, 'dataset_folder', None)
        if dataset:
            candidates.append(os.path.join(os.path.dirname(dataset), DEFAULT_CSV_FILENAME))
        project_folder = getattr(parent, 'project_folder', None)
        if project_folder:
            candidates.append(os.path.join(project_folder, 'data', DEFAULT_CSV_FILENAME))
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return candidates[0] if candidates else None


def _import_merge_geojson():
    try:
        from chronam.merge import merge_geojson as _merge
    except Exception as exc:  # pragma: no cover - import-time failure surfaced to UI
        raise RuntimeError('Add Geographic Info requires geopandas/shapely. Please install the geospatial stack.') from exc
    return _merge


def _import_plot_bar():
    try:
        from chronam.visualize import plot_bar as _plot
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('Collocation bar charts require matplotlib. Install it to view charts.') from exc
    return _plot


def _import_plot_rank_changes():
    try:
        from chronam.visualize import plot_rank_changes as _plot
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('Collocation rank charts require matplotlib. Install it to view charts.') from exc
    return _plot


def _import_plot_articles_by_year():
    try:
        from chronam.visualize import plot_articles_by_year as _plot
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('Yearly article charts require matplotlib. Install it to view charts.') from exc
    return _plot


def _import_plot_topics_over_time():
    try:
        from chronam.visualize import plot_topics_over_time as _plot
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('Topic trend charts require matplotlib. Install it to view charts.') from exc
    return _plot


def _summarize_geojson_outputs(out_paths: List[str]):
    total_articles = 0
    places_all = set()
    html_lines = []
    for path in out_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                geo = json.load(f)
            features = geo.get('features', [])
            total_articles += len(features)
            for feat in features:
                props = feat.get('properties', {})
                places_all.add((props.get('Title'), props.get('SN')))
            encoded = urllib.parse.quote(path)
            html_lines.append(
                f'<div>Output GeoJSON: <a href="chronam-open:{encoded}">{html.escape(path)}</a></div>'
            )
        except Exception:
            continue
    return total_articles, places_all, html_lines


def append_geojson_project_log(parent: 'MainWindow', out_paths: List[str]):  # type: ignore
    total_articles, places_all, html_lines = _summarize_geojson_outputs(out_paths)
    if not html_lines:
        html_lines = ['<div>No GeoJSON files created.</div>']
    stats = getattr(out_paths, 'stats', None)
    summary_lines = []
    if stats:
        total_articles_stat = stats.get("total_articles", total_articles)
        matched_lccn = stats.get("matched_lccn", 0)
        matched_title = stats.get("matched_title", 0)
        matched_total = matched_lccn + matched_title
        total_base = total_articles_stat or (matched_total + stats.get("unmatched", 0)) or 1
        pct = lambda count: (count / total_base * 100.0) if total_base else 0.0
        summary_lines.append(
            f'<div>Added geographic info for {matched_total:,} articles across {len(places_all):,} locations.</div>'
        )
        summary_lines.append(
            f'<div>Matched {matched_lccn:,} articles via LCCN ({pct(matched_lccn):.2f}%) and '
            f'{matched_title:,} via title/date fallback ({pct(matched_title):.2f}%).</div>'
        )
        unmatched_total = stats.get("unmatched", max(total_articles_stat - matched_total, 0))
        summary_lines.append(
            f'<div>{unmatched_total:,} articles had no geographic match ({pct(unmatched_total):.2f}%).</div>'
        )
        unmatched_path = stats.get("unmatched_path")
        if unmatched_path:
            encoded = urllib.parse.quote(unmatched_path)
            summary_lines.append(
                f'<div>Unmatched table: <a href="chronam-open:{encoded}">{html.escape(unmatched_path)}</a></div>'
            )
        summary_lines.append(
            f'<div>Total articles processed: {total_articles_stat:,}</div>'
        )
    else:
        summary_lines.append(
            f'<div>Added geographic info for {total_articles:,} articles across {len(places_all):,} locations.</div>'
        )
    parent.append_project_log('Add Geographic Info', summary_lines + html_lines)


def _default_map_settings() -> dict:
    """Return a fresh copy of the map rendering defaults."""
    return {
        'mode': 'points',
        'time_unit': 'month',
        'time_step': 1,
        'linger_unit': 'month',
        'linger_step': 2,
        'disable_time': False,
        'heat_radius': 15,
        'heat_value': 1.0,
        'grad_min_radius': 6,
        'grad_max_radius': 28,
        'metric': 'article_count',
        'normalize': False,
        'normalize_denominator': 'article_count',
        'lightweight': False,
        'table_mode': 'full',
        'table_row_limit': 0,
    }


def _load_map_settings(raw: Optional[dict]) -> dict:
    """Overlay any persisted map settings onto the defaults."""
    defaults = _default_map_settings()
    if isinstance(raw, dict):
        for key in defaults:
            if key in raw:
                defaults[key] = raw[key]
    return defaults


class Spinner(QWidget):
    def __init__(self, parent=None, radius=20, line_width=4):
        super().__init__(parent)
        self._angle = 0
        self._radius = radius
        self._line_width = line_width
        size = radius * 2 + line_width
        self.setFixedSize(size, size)

    def getAngle(self):
        return self._angle

    def setAngle(self, angle):
        self._angle = angle
        self.update()

    angle = pyqtProperty(int, fget=getAngle, fset=setAngle)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.palette().highlight().color(), self._line_width)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        rect = self.rect().adjusted(
            self._line_width//2, self._line_width//2,
            -self._line_width//2, -self._line_width//2
        )
        painter.drawArc(rect, (self._angle * 16), 270 * 16)

class WorkerThread(QThread):
    progress = pyqtSignal(object)
    finished = pyqtSignal(object)

    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.task_func(*self.args, progress_callback=self.progress.emit, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(e)


class OperationOverlay(QWidget):
    cancel_requested = pyqtSignal()

    def __init__(self, parent: QWidget, message: str):
        super().__init__(parent)
        self._parent = parent
        self._start_time: Optional[float] = None
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._update_elapsed)

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 150);")
        self.setVisible(False)
        self.setFocusPolicy(Qt.NoFocus)
        if parent is not None:
            parent.installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)

        container = QFrame(self)
        container.setStyleSheet("background-color: #ffffff; border-radius: 10px;")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(24, 24, 24, 24)
        container_layout.setSpacing(12)
        container_layout.setAlignment(Qt.AlignCenter)

        self._spinner = Spinner(container, radius=18, line_width=4)
        container_layout.addWidget(self._spinner, 0, Qt.AlignHCenter)

        self._message_label = QLabel(message, container)
        self._message_label.setStyleSheet("font-weight: 600; font-size: 14px;")
        self._message_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self._message_label, 0, Qt.AlignHCenter)

        self._elapsed_label = QLabel("Elapsed: 0s", container)
        self._elapsed_label.setStyleSheet("color: #555555; font-size: 12px;")
        self._elapsed_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self._elapsed_label, 0, Qt.AlignHCenter)

        self._cancel_btn = QPushButton("Cancel", container)
        self._cancel_btn.setFixedWidth(120)
        self._cancel_btn.clicked.connect(self._handle_cancel_clicked)
        container_layout.addWidget(self._cancel_btn, 0, Qt.AlignHCenter)

        layout.addWidget(container, 0, Qt.AlignCenter)

    def eventFilter(self, obj, event):
        if obj is self._parent and event.type() in (QEvent.Resize, QEvent.Move):
            self._sync_geometry()
        return super().eventFilter(obj, event)

    def _sync_geometry(self):
        if self._parent is not None:
            self.setGeometry(self._parent.rect())

    def show_overlay(self):
        self._sync_geometry()
        self._start_time = time.monotonic()
        self._timer.start()
        self.setVisible(True)
        self.raise_()

    def close_overlay(self):
        self._timer.stop()
        self.setVisible(False)
        if self._parent is not None:
            self._parent.removeEventFilter(self)
        self.deleteLater()

    def mark_cancelled(self):
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText("Cancelling…")

    def set_message(self, message: str):
        self._message_label.setText(message)

    def _update_elapsed(self):
        if self._start_time is None:
            self._elapsed_label.setText("Elapsed: 0s")
            return
        elapsed = int(max(0.0, time.monotonic() - self._start_time))
        minutes, seconds = divmod(elapsed, 60)
        if minutes:
            self._elapsed_label.setText(f"Elapsed: {minutes}m {seconds:02d}s")
        else:
            self._elapsed_label.setText(f"Elapsed: {seconds}s")

    def _handle_cancel_clicked(self):
        self.mark_cancelled()
        self.cancel_requested.emit()


class CancelableWorker(QThread):
    finished = pyqtSignal(object, bool)  # result, cancelled
    error = pyqtSignal(Exception)

    def __init__(self, task: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self._task = task
        self._args = args
        self._kwargs = kwargs
        self._cancel_event = threading.Event()

    def run(self):
        try:
            result = self._task(*self._args, cancel_event=self._cancel_event, **self._kwargs)
        except OperationCancelledError:
            self.finished.emit(None, True)
            return
        except Exception as exc:
            self.error.emit(exc)
            return
        if self._cancel_event.is_set():
            self.finished.emit(result, True)
        else:
            self.finished.emit(result, False)

    def request_cancel(self):
        self._cancel_event.set()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

class CloseShortcutFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_W:
            modifiers = event.modifiers()
            if modifiers & (Qt.ControlModifier | Qt.MetaModifier):
                window = QApplication.activeWindow()
                if window is not None and hasattr(window, 'close'):
                    window.close()
                    return True
        return super().eventFilter(obj, event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._base_title = 'Untitled'
        self.setWindowTitle(self._base_title)
        self.resize(900, 600)
        self.project_folder = os.getcwd()
        self.dataset_folder = None
        self.dataset_years = []
        self.json_file = None
        self.geojson_file = None
        self.locations_csv_path = None
        self.search_log_history = []
        self.project_log_entries = []
        self.project_file = None
        self.collocation_state = {'dropped_terms': [], 'term_groups': [], 'topic_settings': {}, 'topic_trend_settings': {}}
        self.collocation_drop_terms = []
        self.collocation_term_groups: List[dict] = []
        self.map_settings = _default_map_settings()
        self.metadata_enabled = True
        self.init_ui()
        self._close_filter = CloseShortcutFilter()
        QApplication.instance().installEventFilter(self._close_filter)
        self._update_window_title()

    def init_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        new_project_action = self._action('New Project', self.new_project)
        file_menu.addAction(new_project_action)

        open_project_action = self._action('Open Project', self.open_project)
        open_project_action.setShortcut(QKeySequence.Open)
        file_menu.addAction(open_project_action)

        save_project_action = self._action('Save Project', self.save_project)
        save_project_action.setShortcut(QKeySequence.Save)
        file_menu.addAction(save_project_action)

        file_menu.addAction(self._action('Save Project As...', self.save_project_as))

        quit_action = self._action('Quit', self._handle_quit_request)
        quit_action.setMenuRole(QAction.NoRole)
        quit_action.setShortcut(QKeySequence.Quit)
        file_menu.addAction(quit_action)

        sources_menu = menubar.addMenu('Sources')
        sources_menu.addAction(self._action('Set Local Dataset Folder...', self.set_dataset_folder))
        sources_menu.addAction(self._action('Set Newspaper Locations Table...', self.set_locations_table))
        sources_menu.addAction(self._action('Open Project Folder in Finder', self.open_project_folder))

        view_menu = menubar.addMenu('View')
        view_menu.addAction(self._action('Project Log', self.show_project_log))

        self.container = QWidget()
        self.setCentralWidget(self.container)
        main_layout = QVBoxLayout(self.container)

        # Load JSON/GeoJSON section
        load_layout = QHBoxLayout()
        self.load_json_btn = QPushButton('Load JSON')
        self.load_json_btn.clicked.connect(self.open_json_file)
        self.json_label = QLabel('No JSON loaded')
        load_layout.addWidget(self.load_json_btn)
        load_layout.addWidget(self.json_label)
        self.load_geojson_btn = QPushButton('Load GeoJSON')
        self.load_geojson_btn.clicked.connect(self.open_geojson_file)
        self.geojson_label = QLabel('No GeoJSON loaded')
        load_layout.addWidget(self.load_geojson_btn)
        load_layout.addWidget(self.geojson_label)
        main_layout.addLayout(load_layout)

        main_layout.addWidget(QLabel("Select an action below:"))

        self.btn_download = QPushButton('A) Search Dataset')
        self.btn_download.clicked.connect(self.action_download)
        self.btn_update = QPushButton('B) Add Geographic Info')
        self.btn_update.clicked.connect(self.action_update_locations)
        self.btn_collocate = QPushButton('C) Text Analysis')
        self.btn_collocate.clicked.connect(self.action_collocate)
        self.btn_map = QPushButton('D) Create Map')
        self.btn_map.clicked.connect(self.open_create_map_dialog)
        for btn in (self.btn_download, self.btn_update, self.btn_collocate, self.btn_map):
            main_layout.addWidget(btn)

        self.metadata_checkbox = QCheckBox('Create metadata JSON for all output files')
        self.metadata_checkbox.setChecked(self.metadata_enabled)
        self.metadata_checkbox.stateChanged.connect(self._on_metadata_checkbox_toggled)
        main_layout.addWidget(self.metadata_checkbox)

        self._init_project_log()
        self._update_primary_button_styles()

    def _action(self, name, slot):
        a = QAction(name, self)
        a.triggered.connect(slot)
        return a


    def _init_project_log(self):
        self.project_log_browser = QTextBrowser()
        self.project_log_browser.setReadOnly(True)
        self.project_log_browser.setOpenLinks(False)
        self.project_log_browser.anchorClicked.connect(self._handle_project_log_link)

        dock = QDockWidget('Project Log', self)
        dock.setObjectName('ProjectLogDock')
        dock.setWidget(self.project_log_browser)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.project_log_dock = dock
        self._refresh_project_log_widget()

    def show_project_log(self):
        if hasattr(self, 'project_log_dock'):
            self.project_log_dock.show()
            self.project_log_dock.raise_()

    def _handle_project_log_link(self, url: QUrl):
        if url.scheme() != 'chronam-open':
            return
        encoded = url.toString()[len('chronam-open:'):]
        path = urllib.parse.unquote(encoded)
        reveal_in_file_manager(path)

    def _refresh_project_log_widget(self):
        if not hasattr(self, 'project_log_browser'):
            return
        self.project_log_browser.clear()
        for entry in self.project_log_entries:
            self.project_log_browser.append(entry)
        self.project_log_browser.moveCursor(QTextCursor.End)

    def _handle_quit_request(self):
        self.close()

    def closeEvent(self, event):
        if self._confirm_quit():
            event.accept()
            super().closeEvent(event)
        else:
            event.ignore()

    def _confirm_quit(self) -> bool:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Question)
        dialog.setWindowTitle('Quit ChronAM Project')
        dialog.setText('Are you sure you want to quit?')
        save_button = dialog.addButton('Save and Quit', QMessageBox.AcceptRole)
        quit_button = dialog.addButton('Quit without Saving', QMessageBox.DestructiveRole)
        cancel_button = dialog.addButton(QMessageBox.Cancel)
        if self.project_file:
            dialog.setDefaultButton(save_button)
        else:
            dialog.setDefaultButton(quit_button)
        dialog.exec_()
        clicked = dialog.clickedButton()
        if clicked == cancel_button:
            return False
        if clicked == save_button:
            had_project_path = bool(self.project_file)
            self.save_project()
            if not had_project_path and not self.project_file:
                return False
            return True
        return True

    def set_locations_table(self):
        start_dir = self.locations_csv_path or os.path.join(self.project_folder, 'data')
        if not (start_dir and os.path.isdir(start_dir)):
            start_dir = self.project_folder or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Select Newspaper Locations CSV',
            start_dir,
            'CSV Files (*.csv)'
        )
        if not path:
            return
        self.locations_csv_path = path
        self.append_project_log('Sources', [f'<div>Locations CSV set to: {html.escape(path)}</div>'])

    def open_project_folder(self):
        folder = self.project_folder
        if not (folder and os.path.isdir(folder)):
            QMessageBox.warning(self, 'Project Folder Missing', 'Project folder is not available to open.')
            return
        reveal_in_file_manager(folder)

    def append_project_log(self, tool_name: str, html_lines: list):
        if not html_lines:
            return
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header = html.escape(tool_name)
        entry_parts = [
            '<hr/><hr/>',
            f'<div><strong>{header}</strong> — {timestamp}</div>'
        ]
        entry_parts.extend(self._format_project_log_lines(html_lines))
        entry_html = ''.join(entry_parts)
        self.project_log_entries.append(entry_html)
        if hasattr(self, 'project_log_browser'):
            self.project_log_browser.append(entry_html)
            self.project_log_browser.moveCursor(QTextCursor.End)

    def _format_project_log_lines(self, html_lines: list) -> list:
        formatted = []
        for raw in html_lines:
            if raw is None:
                continue
            raw = str(raw)
            stripped = raw.strip()
            if not stripped:
                formatted.append('<br/>')
            elif stripped.startswith('<div') or stripped.startswith('<hr') or stripped.startswith('<br'):
                formatted.append(raw)
            else:
                formatted.append(f'<div>{raw}</div>')
        return formatted


    def _project_display_name(self):
        if self.project_file:
            return os.path.splitext(os.path.basename(self.project_file))[0]
        return 'Untitled'

    def _update_window_title(self):
        self.setWindowTitle(self._project_display_name())

    def _update_loaded_file_labels(self):
        if hasattr(self, 'json_label'):
            if self.json_file and os.path.exists(self.json_file):
                self.json_label.setText(os.path.basename(self.json_file))
            elif self.json_file:
                self.json_label.setText(f"(missing) {os.path.basename(self.json_file)}")
            else:
                self.json_label.setText('No JSON loaded')

        if hasattr(self, 'geojson_label'):
            if self.geojson_file and os.path.exists(self.geojson_file):
                self.geojson_label.setText(os.path.basename(self.geojson_file))
            elif self.geojson_file:
                self.geojson_label.setText(f"(missing) {os.path.basename(self.geojson_file)}")
            else:
                self.geojson_label.setText('No GeoJSON loaded')

        self._update_primary_button_styles()

    def _update_primary_button_styles(self):
        has_json = bool(self.json_file and os.path.exists(self.json_file))
        has_geo = bool(self.geojson_file and os.path.exists(self.geojson_file))

        buttons = {
            'download': getattr(self, 'btn_download', None),
            'load_json': getattr(self, 'load_json_btn', None),
            'load_geo': getattr(self, 'load_geojson_btn', None),
            'update': getattr(self, 'btn_update', None),
            'collocate': getattr(self, 'btn_collocate', None),
            'map': getattr(self, 'btn_map', None),
        }

        highlight_states = {
            buttons['download']: not has_json and not has_geo,
            buttons['load_json']: not has_json and not has_geo,
            buttons['load_geo']: not has_json and not has_geo,
            buttons['update']: has_json and not has_geo,
            buttons['collocate']: (has_json and not has_geo) or has_geo,
            buttons['map']: has_geo,
        }

        for btn, highlight in highlight_states.items():
            if btn is None:
                continue
            if highlight:
                btn.setStyleSheet(self._search_tool_highlight_style())
            else:
                btn.setStyleSheet('')

    def _on_metadata_checkbox_toggled(self, state: int):
        self.metadata_enabled = state == Qt.Checked

    @staticmethod
    def _search_tool_highlight_style() -> str:
        return (
            "QPushButton {"
            " background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2b8cff, stop:1 #0066ff);"
            " color: #ffffff;"
            " border: 1px solid #0060e0;"
            " border-radius: 8px;"
            "}"
            "QPushButton:pressed {"
            " background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e7af2, stop:1 #0051cc);"
            " border: 1px solid #004bbd;"
            "}"
        )

    def new_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Create Project',
            os.path.join(self.project_folder or os.getcwd(), 'NewProject'),
            'ChronAM Project Folder (*)'
        )
        if not path:
            return

        path = os.path.splitext(path)[0]
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, 'data', 'raw'), exist_ok=True)
        os.makedirs(os.path.join(path, 'data', 'processed'), exist_ok=True)

        self.project_folder = path
        self.project_file = os.path.join(path, 'chronam_project.chronam.json')
        self.dataset_folder = None
        self.dataset_years = []
        self.json_file = None
        self.geojson_file = None
        self.locations_csv_path = None
        self.search_log_history.clear()
        self.project_log_entries.clear()
        self.collocation_state = {'dropped_terms': [], 'term_groups': [], 'topic_settings': {}, 'topic_trend_settings': {}}
        self.collocation_drop_terms = []
        self.collocation_term_groups = []
        self.map_settings = _default_map_settings()
        self.metadata_enabled = True
        if hasattr(self, 'metadata_checkbox'):
            self.metadata_checkbox.blockSignals(True)
            self.metadata_checkbox.setChecked(True)
            self.metadata_checkbox.blockSignals(False)

        self.ensure_dataset_folder(prompt=False)
        self._update_loaded_file_labels()
        self._refresh_project_log_widget()
        if self.project_file:
            self._write_project_file(self.project_file)
        self._update_window_title()
        self.append_project_log('Project', [f'<div>New project created at: {html.escape(path)}</div>'])

    def open_project(self):
        start_dir = self.project_file or os.path.join(self.project_folder or os.getcwd(), 'chronam_project.json')
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Open Project File',
            start_dir,
            'ChronAM Project (*.chronam.json *.json);;All Files (*)'
        )
        if not path:
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, 'Open Project Failed', f'Unable to load project file:\n{exc}')
            return

        self.project_file = path
        self.project_folder = data.get('project_folder') or os.path.dirname(path)

        self.dataset_folder = data.get('dataset_folder')
        stored_years = data.get('dataset_years', [])
        self.dataset_years = []
        if self.dataset_folder and not self._apply_dataset_folder(self.dataset_folder):
            # keep stored path even if parquet files are missing
            self.dataset_years = stored_years if isinstance(stored_years, list) else []

        self.json_file = data.get('json_file')
        self.geojson_file = data.get('geojson_file')
        locations_csv = data.get('locations_csv_path')
        self.locations_csv_path = locations_csv if isinstance(locations_csv, str) else None
        collocation_state = data.get('collocation_state')
        self.collocation_state = dict(collocation_state) if isinstance(collocation_state, dict) else {}
        self.map_settings = _load_map_settings(data.get('map_settings'))
        drop_terms = data.get('collocation_drop_terms')
        if drop_terms is None:
            drop_terms = self.collocation_state.get('dropped_terms') if isinstance(self.collocation_state, dict) else []
        if isinstance(drop_terms, list):
            self.collocation_drop_terms = [str(term) for term in drop_terms if isinstance(term, str) and term.strip()]
        else:
            self.collocation_drop_terms = []
        if isinstance(self.collocation_state, dict):
            self.collocation_state['dropped_terms'] = list(self.collocation_drop_terms)

        groups_value = data.get('collocation_term_groups')
        if groups_value is None and isinstance(self.collocation_state, dict):
            groups_value = self.collocation_state.get('term_groups')
        self.collocation_term_groups = []
        if isinstance(groups_value, list):
            for entry in groups_value:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get('name', '')).strip()
                if not name:
                    continue
                terms_raw = entry.get('terms') or []
                terms: List[str] = []
                seen_terms: Set[str] = set()
                for term in terms_raw:
                    term_str = str(term).strip()
                    if not term_str:
                        continue
                    lower = term_str.lower()
                    if lower in seen_terms:
                        continue
                    seen_terms.add(lower)
                    terms.append(term_str)
                if not terms:
                    continue
                group_record: Dict[str, Any] = {'name': name, 'terms': terms}
                freq_val = entry.get('total_frequency')
                try:
                    if freq_val is not None:
                        group_record['total_frequency'] = float(freq_val)
                except (TypeError, ValueError):
                    pass
                missing_terms = entry.get('missing_terms')
                if isinstance(missing_terms, list):
                    cleaned_missing = [str(term).strip() for term in missing_terms if str(term).strip()]
                    if cleaned_missing:
                        group_record['missing_terms'] = cleaned_missing
                self.collocation_term_groups.append(group_record)
        if isinstance(self.collocation_state, dict):
            self.collocation_state['term_groups'] = [dict(group) for group in self.collocation_term_groups]

        self.metadata_enabled = bool(data.get('metadata_enabled', True))
        if hasattr(self, 'metadata_checkbox'):
            self.metadata_checkbox.blockSignals(True)
            self.metadata_checkbox.setChecked(self.metadata_enabled)
            self.metadata_checkbox.blockSignals(False)

        search_log = data.get('search_log_history')
        if search_log is None:
            search_log = data.get('download_log', [])
        if isinstance(search_log, list):
            self.search_log_history = list(search_log)
        else:
            self.search_log_history = []

        project_log = data.get('project_log')
        if project_log is None:
            project_log = data.get('project_log_entries', [])
        if isinstance(project_log, list):
            self.project_log_entries = list(project_log)
        else:
            self.project_log_entries = []

        QMessageBox.information(self, 'Project Loaded', f'Loaded project:\n{path}')
        self._update_loaded_file_labels()
        self._refresh_project_log_widget()
        self._update_window_title()

    def save_project(self):
        if not self.project_file:
            self.save_project_as()
            return
        if self._write_project_file(self.project_file):
            QMessageBox.information(self, 'Project Saved', f'Project saved to:\n{self.project_file}')
            self._update_window_title()

    def save_project_as(self):
        start_dir = self.project_file or os.path.join(self.project_folder or os.getcwd(), 'chronam_project.json')
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Project As',
            start_dir,
            'ChronAM Project (*.chronam.json *.json);;All Files (*)'
        )
        if not path:
            return
        if self._write_project_file(path):
            self.project_file = path
            QMessageBox.information(self, 'Project Saved', f'Project saved to:\n{path}')
            self._update_window_title()

    def _write_project_file(self, path: str) -> bool:
        data = {
            'version': 1,
            'project_folder': self.project_folder,
            'dataset_folder': self.dataset_folder,
            'dataset_years': self.dataset_years,
            'json_file': self.json_file,
            'geojson_file': self.geojson_file,
            'locations_csv_path': self.locations_csv_path,
            'search_log_history': self.search_log_history,
            'project_log': self.project_log_entries,
            'map_settings': dict(self.map_settings),
            'collocation_state': dict(self.collocation_state),
            'collocation_drop_terms': list(self.collocation_drop_terms),
            'collocation_term_groups': [dict(group) for group in self.collocation_term_groups],
            'metadata_enabled': bool(self.metadata_enabled),
        }

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, 'Save Project Failed', f'Unable to save project file:\n{exc}')
            return False
        return True

    def set_dataset_folder(self):
        start_dir = self.dataset_folder or self.project_folder
        folder = QFileDialog.getExistingDirectory(self, 'Select Local Dataset Folder', start_dir)
        if folder:
            if not self._apply_dataset_folder(folder):
                QMessageBox.warning(self, 'Dataset Folder Required', DATASET_FOLDER_WARNING)
                return
            self._update_loaded_file_labels()
            self._update_window_title()
            self._refresh_project_log_widget()
            self.append_project_log('Sources', [f'<div>Dataset folder set to: {html.escape(folder)}</div>'])

    def _dataset_folder_candidates(self):
        seen = set()
        for path in (
            getattr(self, 'dataset_folder', None),
            os.path.join(self.project_folder, 'data', 'parquet'),
            os.path.join(self.project_folder, 'parquet'),
        ):
            if path and path not in seen:
                seen.add(path)
                yield path

    def _discover_dataset_years(self, folder: str):
        if not folder or not os.path.isdir(folder):
            return []
        try:
            years = []
            pattern = re.compile(r"AmericanStories_(\d{4})\.parquet$")
            for name in os.listdir(folder):
                match = pattern.match(name)
                if match:
                    years.append(int(match.group(1)))
            return sorted(set(years))
        except OSError:
            return []

    def _apply_dataset_folder(self, folder: str) -> bool:
        years = self._discover_dataset_years(folder)
        if years:
            self.dataset_folder = folder
            self.dataset_years = years
            return True
        return False

    def ensure_dataset_folder(self, prompt: bool = True):
        self.dataset_years = []
        for candidate in self._dataset_folder_candidates():
            if self._apply_dataset_folder(candidate):
                return self.dataset_folder

        self.dataset_folder = None

        if not prompt:
            return None

        QMessageBox.warning(
            self,
            'Dataset Folder Required',
            DATASET_FOLDER_WARNING
        )
        folder = QFileDialog.getExistingDirectory(self, 'Select Local Dataset Folder', self.project_folder)
        if folder and self._apply_dataset_folder(folder):
            return self.dataset_folder
        if folder:
            QMessageBox.warning(self, 'Dataset Folder Required', DATASET_FOLDER_WARNING)
        return None

    def open_json_file(self):
        start_dir = os.path.join(self.project_folder, 'data', 'processed') if self.project_folder else ''
        if not (start_dir and os.path.isdir(start_dir)):
            start_dir = self.project_folder or ''
        file_path, _ = QFileDialog.getOpenFileName(self, 'Load JSON File', start_dir, 'JSON Files (*.json)')
        if file_path:
            self.json_file = file_path
            self._update_loaded_file_labels()

    def open_geojson_file(self):
        start_dir = os.path.join(self.project_folder, 'data', 'processed') if self.project_folder else ''
        if not (start_dir and os.path.isdir(start_dir)):
            start_dir = self.project_folder or ''
        file_path, _ = QFileDialog.getOpenFileName(self, 'Load GeoJSON File', start_dir, 'GeoJSON Files (*.geojson *.json)')
        if file_path:
            self.geojson_file = file_path
            self._update_loaded_file_labels()

    def action_download(self):
        dlg = DownloadDialog(self)
        dlg.show()

    def action_update_locations(self):
        dlg = UpdateLocationsDialog(self)
        dlg.exec_()

    def action_collocate(self):
        dlg = CollocationDialog(self)
        dlg.setModal(False)
        dlg.setWindowModality(Qt.NonModal)
        dlg.show()


    def open_create_map_dialog(self):
        dlg = MapToolDialog(self)
        dlg.setModal(False)
        dlg.setWindowModality(Qt.NonModal)
        dlg.show()

class DownloadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Search Dataset')
        self.setMinimumSize(600, 400)
        self._csv_hint_shown = False

        layout = QVBoxLayout(self)

        dataset_row = QHBoxLayout()
        self.dataset_label = QLabel()
        self.dataset_label.setWordWrap(True)
        dataset_row.addWidget(self.dataset_label, 1)
        self.dataset_change_btn = QPushButton('Change')
        self.dataset_change_btn.clicked.connect(self._change_dataset_folder)
        dataset_row.addWidget(self.dataset_change_btn, 0)
        layout.addLayout(dataset_row)

        self.dataset_years_label = QLabel()
        self.dataset_years_label.setWordWrap(True)
        layout.addWidget(self.dataset_years_label)

        self.spinner = Spinner(self)
        self.spinner.hide()
        layout.addWidget(self.spinner, alignment=Qt.AlignCenter)

        self._anim = QPropertyAnimation(self.spinner, b"angle", self)
        self._anim.setStartValue(0)
        self._anim.setEndValue(360)
        self._anim.setDuration(1000)
        self._anim.setLoopCount(-1)

        form = QFormLayout()
        self.search_input = QLineEdit()
        self.start_input = QLineEdit()
        self.end_input = QLineEdit()
        form.addRow('Search Term:', self.search_input)
        form.addRow('Start Date (YYYY-MM-DD):', self.start_input)
        form.addRow('End Date (YYYY-MM-DD):', self.end_input)
        layout.addLayout(form)

        cleaning_group = QGroupBox('Text Cleaning Options')
        cleaning_layout = QVBoxLayout(cleaning_group)
        self.clean_lowercase_cb = QCheckBox('Convert article text to lowercase')
        self.clean_urls_cb = QCheckBox('Change article URLs to end in .pdf (replaces .jp2)')
        self.clean_urls_cb.setChecked(True)
        self.clean_hyphen_cb = QCheckBox('Collapse hyphenated breaks (remove "- " sequences)')
        self.clean_geo_cb = QCheckBox('Add Geographic Info (create GeoJSON output)')
        self.clean_geo_unmatched_cb = QCheckBox('Create table of unmatched articles')
        self.clean_geo_unmatched_cb.setEnabled(False)
        self.clean_geo_cb.toggled.connect(self.clean_geo_unmatched_cb.setEnabled)
        self.clean_geo_cb.toggled.connect(lambda checked: None if checked else self.clean_geo_unmatched_cb.setChecked(False))
        self.clean_geo_cb.setChecked(True)
        cleaning_layout.addWidget(self.clean_lowercase_cb)
        cleaning_layout.addWidget(self.clean_urls_cb)
        cleaning_layout.addWidget(self.clean_hyphen_cb)
        cleaning_layout.addWidget(self.clean_geo_cb)
        cleaning_layout.addWidget(self.clean_geo_unmatched_cb)
        layout.addWidget(cleaning_group)

        outputs_group = QGroupBox('Summary Outputs')
        outputs_layout = QVBoxLayout(outputs_group)
        outputs_layout.setContentsMargins(9, 9, 9, 9)
        self.yearly_csv_cb = QCheckBox('Create yearly summary CSV')
        self.yearly_csv_only_cb = QCheckBox('Create yearly summary CSV (only)')
        self.yearly_chart_cb = QCheckBox('Show yearly article chart')
        self.yearly_chart_cb.setChecked(True)
        outputs_layout.addWidget(self.yearly_csv_cb)
        outputs_layout.addWidget(self.yearly_csv_only_cb)
        outputs_layout.addWidget(self.yearly_chart_cb)
        layout.addWidget(outputs_group)

        self.log = QTextBrowser()
        self.log.setOpenLinks(False)
        self.log.anchorClicked.connect(self._handle_log_link)
        self.log.setReadOnly(True)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        layout.addWidget(self.log)
        layout.addWidget(self.progress)
        self.progress_details = QLabel()
        self.progress_details.setWordWrap(True)
        self.progress_details.setStyleSheet("color:#555; font-size:11px;")
        self.progress_details.hide()
        layout.addWidget(self.progress_details)
        self.progress_eta = QLabel()
        self.progress_eta.setWordWrap(True)
        self.progress_eta.setStyleSheet("color:#555; font-size:11px;")
        self.progress_eta.hide()
        layout.addWidget(self.progress_eta)

        btns = QHBoxLayout()
        self.run_btn = QPushButton('Search Records')
        self.run_btn.clicked.connect(self.start_download)
        self.cancel_btn = QPushButton('Close')
        self.cancel_btn.clicked.connect(self.cancel_download)
        btns.addWidget(self.run_btn)
        btns.addWidget(self.cancel_btn)
        layout.addLayout(btns)

        self.thread = None
        self._start_time = None

        self.logged_years = set()
        self.current_parquet_dir = None
        self._search_running = False
        self._cancel_event = threading.Event()
        self._cancel_requested = False
        self._year_timer = None

        self._log_history = parent.search_log_history
        self._restore_log_history()
        self.refresh_dataset_label()
        self._current_run_lines = []

        self._current_term = ''
        self._current_start = ''
        self._current_end = ''
        self._current_term_dir: Optional[str] = None
        self._summary_only_active = False
        self._summary_only_requested = False
        self._current_start_date_obj = None
        self._current_end_date_obj = None
        self._progress_metric_key = 'article_count'
        self._progress_total_metric = 0
        self._progress_cumulative = 0
        self._progress_year_totals: Dict[str, int] = {}
        self._processed_dataset_units = 0
        self._processed_years: Set[str] = set()
        self._first_year_sec_per_unit: Optional[float] = None
        self._eta_seconds_remaining: Optional[float] = None

        self.yearly_csv_only_cb.toggled.connect(self._update_summary_outputs_state)
        self.yearly_csv_cb.toggled.connect(self._update_summary_outputs_state)
        self.yearly_chart_cb.toggled.connect(self._update_summary_outputs_state)
        self._update_summary_outputs_state()

    def showEvent(self, event):
        self.refresh_dataset_label()
        super().showEvent(event)

    def refresh_dataset_label(self):
        folder = getattr(self.parent(), 'dataset_folder', None)
        years = getattr(self.parent(), 'dataset_years', [])
        if folder and os.path.isdir(folder) and years:
            folder_text = f"Dataset folder: {folder}"
        elif folder and os.path.isdir(folder):
            folder_text = f"Dataset folder: {folder} (no AmericanStories parquet files found)"
        else:
            folder_text = "Dataset folder: Not set"
        self.dataset_label.setText(folder_text)
        self.dataset_change_btn.setEnabled(True)
        self.dataset_years_label.setText(self._format_year_summary(years))
        self._apply_year_defaults(years)

    def _update_summary_outputs_state(self):
        summary_only = self.yearly_csv_only_cb.isChecked()
        if summary_only and not self._summary_only_active:
            self._prev_yearly_chart_state = self.yearly_chart_cb.isChecked()
            self._prev_clean_geo_state = self.clean_geo_cb.isChecked()
        if summary_only:
            self._summary_only_active = True
            if not self.yearly_csv_cb.isChecked():
                self.yearly_csv_cb.blockSignals(True)
                self.yearly_csv_cb.setChecked(True)
                self.yearly_csv_cb.blockSignals(False)
            self.yearly_csv_cb.setEnabled(False)
            if self.yearly_chart_cb.isChecked():
                self.yearly_chart_cb.blockSignals(True)
                self.yearly_chart_cb.setChecked(False)
                self.yearly_chart_cb.blockSignals(False)
            self.yearly_chart_cb.setEnabled(False)
            if self.clean_geo_cb.isChecked():
                self.clean_geo_cb.blockSignals(True)
                self.clean_geo_cb.setChecked(False)
                self.clean_geo_cb.blockSignals(False)
            self.clean_geo_cb.setEnabled(False)
        else:
            if self._summary_only_active:
                self.yearly_chart_cb.setChecked(getattr(self, '_prev_yearly_chart_state', True))
                if getattr(self, '_prev_clean_geo_state', False):
                    self.clean_geo_cb.setChecked(True)
            self._summary_only_active = False
            self.yearly_csv_cb.setEnabled(True)
            self.yearly_chart_cb.setEnabled(True)
            self.clean_geo_cb.setEnabled(True)

        self.clean_geo_unmatched_cb.setEnabled(self.clean_geo_cb.isChecked() and self.clean_geo_cb.isEnabled())

    def _change_dataset_folder(self):
        start_dir = getattr(self.parent(), 'dataset_folder', None) or self.parent().project_folder
        folder = QFileDialog.getExistingDirectory(self, 'Select Local Dataset Folder', start_dir)
        if folder:
            if not self.parent()._apply_dataset_folder(folder):
                QMessageBox.warning(self, 'Dataset Folder Required', DATASET_FOLDER_WARNING)
            else:
                self.refresh_dataset_label()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.ActiveWindowFocusReason)

    def _store_csv_path(self, path: Optional[str]):
        if not (path and os.path.exists(path)):
            return
        parent = self.parent()
        if parent is not None:
            parent.locations_csv_path = path

    def _prompt_csv_path(self) -> Optional[str]:
        parent = self.parent()
        if not self._csv_hint_shown:
            QMessageBox.information(
                self,
                'Locate Locations CSV',
                f'Locate the newspaper locations table named "{DEFAULT_CSV_FILENAME}".'
            )
            self._csv_hint_shown = True
        if parent and getattr(parent, 'project_folder', None):
            start = resolve_locations_csv(parent)
            if not start or not os.path.isfile(start):
                start = parent.project_folder
        else:
            start = resolve_locations_csv(None) or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            f'Select Locations CSV ({DEFAULT_CSV_FILENAME})',
            start,
            'CSV Files (*.csv)'
        )
        if path:
            self._store_csv_path(path)
        return path or None

    def _ensure_geo_csv_path(self) -> Optional[str]:
        candidate = resolve_locations_csv(self.parent())
        if candidate and os.path.exists(candidate):
            self._store_csv_path(candidate)
            return candidate
        return self._prompt_csv_path()

    def _add_geo_after_search(self, json_path: str) -> List[str]:
        parent = self.parent()
        if parent is None:
            return []
        csv_path = self._ensure_geo_csv_path()
        if not (csv_path and os.path.exists(csv_path)):
            self._log_plain('Skipped geographic info — locations CSV not selected.')
            return []
        unmatched_csv_path = None
        if self.clean_geo_unmatched_cb.isChecked():
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            unmatched_csv_path = os.path.join(
                parent.project_folder,
                'data',
                'processed',
                f'unmatched_{base_name}.csv'
            )
        try:
            merge_geojson = _import_merge_geojson()
            out_paths = merge_geojson(
                parent.project_folder,
                csv_path=csv_path,
                json_path=json_path,
                unmatched_csv_path=unmatched_csv_path
            )
        except Exception as exc:
            QMessageBox.warning(self, 'Add Geographic Info', f'Could not add geographic info: {exc}')
            self._log_plain(f'Adding geographic info failed: {exc}')
            return []

        if out_paths:
            parent.geojson_file = out_paths[-1]
            if parent.locations_csv_path and not os.path.samefile(parent.locations_csv_path, csv_path):
                parent.locations_csv_path = csv_path
            parent.collocation_state = {'dropped_terms': [], 'term_groups': [], 'topic_settings': {}, 'topic_trend_settings': {}}
            parent.collocation_drop_terms = []
            parent._update_loaded_file_labels()
        return out_paths

    def _ensure_log_visible(self):
        self.log.moveCursor(QTextCursor.End)
        self.log.ensureCursorVisible()

    def _restore_log_history(self):
        self.log.clear()
        for entry in self._log_history:
            self.log.append(entry)
        if self._log_history:
            self._ensure_log_visible()

    def _append_log_html(self, html: str):
        self.log.append(html)
        self._log_history.append(html)
        if hasattr(self, '_current_run_lines') and self._current_run_lines is not None:
            self._current_run_lines.append(html)
        self._ensure_log_visible()

    @staticmethod
    def _format_year_summary(years):
        if not years:
            return "Years available: none"

        parts = []
        start = prev = years[0]
        for y in years[1:]:
            if y == prev + 1:
                prev = y
                continue
            parts.append((start, prev))
            start = prev = y
        parts.append((start, prev))

        text_parts = []
        for start, end in parts:
            if start == end:
                text_parts.append(str(start))
            else:
                text_parts.append(f"{start}-{end}")
        return "Years available: " + ", ".join(text_parts)

    def _apply_year_defaults(self, years: List[int]):
        if not years:
            return
        if self.start_input.text().strip() or self.end_input.text().strip():
            return

        sorted_years = sorted(set(years))
        if not sorted_years:
            return

        best_start = best_end = sorted_years[0]
        current_start = current_end = sorted_years[0]

        for year in sorted_years[1:]:
            if year == current_end + 1:
                current_end = year
            else:
                if (current_end - current_start) > (best_end - best_start):
                    best_start, best_end = current_start, current_end
                current_start = current_end = year

        if (current_end - current_start) > (best_end - best_start):
            best_start, best_end = current_start, current_end

        start_date = f"{best_start:04d}-01-01"
        end_date = f"{best_end:04d}-12-31"
        self.start_input.setText(start_date)
        self.end_input.setText(end_date)

    def _log_plain(self, text: str):
        safe = html.escape(text)
        self._append_log_html(f"<span>{safe}</span>")

    def _log_blank(self):
        self._append_log_html('')

    def _log_separator(self):
        self._append_log_html('<hr/>')

    def _log_link(self, prefix: str, path: str, elapsed: Optional[float] = None):
        encoded = urllib.parse.quote(path)
        safe_prefix = html.escape(prefix)
        safe_path = html.escape(path)
        if elapsed is not None:
            message = (
                f"<span>{safe_prefix} {elapsed:.1f}s — saved to "
                f"<a href=\"chronam-open:{encoded}\">{safe_path}</a></span>"
            )
        elif safe_prefix:
            message = (
                f"<span>{safe_prefix}: "
                f"<a href=\"chronam-open:{encoded}\">{safe_path}</a></span>"
            )
        else:
            message = f"<span><a href=\"chronam-open:{encoded}\">{safe_path}</a></span>"
        self._append_log_html(message)

    def _finalize_project_log(self, tool_name='Search Dataset'):
        if getattr(self, '_current_run_lines', None):
            self.parent().append_project_log(tool_name, list(self._current_run_lines))
            self._current_run_lines = []

    def _set_running_state(self, running: bool):
        self._search_running = running
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(True)
        self.dataset_change_btn.setEnabled(not running)
        if running:
            self.cancel_btn.setText('Cancel Search')
            self.spinner.show()
            self._anim.start()
        else:
            self.cancel_btn.setText('Close')
            self._anim.stop()
            self.spinner.hide()
            self.current_parquet_dir = None
            if self.thread and not self.thread.isRunning():
                self.thread = None
            self._year_timer = None

    def _handle_log_link(self, url: QUrl):
        if url.scheme() != 'chronam-open':
            return
        encoded = url.toString()[len('chronam-open:'):]
        path = urllib.parse.unquote(encoded)
        if not path:
            return
        reveal_in_file_manager(path)

    def start_download(self):
        self.refresh_dataset_label()

        term  = self.search_input.text().strip()
        start = self.start_input.text().strip()
        end   = self.end_input.text().strip()
        summary_only = self.yearly_csv_only_cb.isChecked()

        try:
            start_date_obj = datetime.strptime(start, '%Y-%m-%d').date()
            end_date_obj = datetime.strptime(end, '%Y-%m-%d').date()
        except ValueError:
            QMessageBox.warning(self, 'Invalid Dates', 'Enter start and end dates as YYYY-MM-DD.')
            self._log_plain('Search cancelled — invalid date format.')
            self._finalize_project_log()
            return

        if end_date_obj < start_date_obj:
            QMessageBox.warning(self, 'Invalid Range', 'The end date must be on or after the start date.')
            self._log_plain('Search cancelled — end date precedes start date.')
            self._finalize_project_log()
            return

        self._current_run_lines = []
        dataset_folder = self.parent().ensure_dataset_folder()
        if not dataset_folder:
            self._log_plain('Search cancelled — dataset folder not recognized.')
            self._finalize_project_log()
            return

        self.current_parquet_dir = dataset_folder
        self.refresh_dataset_label()

        # Single output path for the full range
        processed_root = os.path.join(self.parent().project_folder, 'data', 'processed')
        term_dir = os.path.join(processed_root, term_directory_name(term))
        os.makedirs(term_dir, exist_ok=True)
        if not summary_only:
            out_path = os.path.join(term_dir, f"{term}_{start}_{end}.json")
            if os.path.exists(out_path):
                if QMessageBox.warning(
                    self, 'Overwrite Warning', f'Will overwrite:\n{out_path}',
                    QMessageBox.Yes | QMessageBox.No
                ) != QMessageBox.Yes:
                    self._log_plain('Search cancelled — existing file retained.')
                    self._finalize_project_log()
                    return

        self._current_term = term
        self._current_start = start
        self._current_end = end
        self._current_term_dir = term_dir
        self._summary_only_requested = summary_only
        self._current_start_date_obj = start_date_obj
        self._current_end_date_obj = end_date_obj

        if self.log.toPlainText().strip():
            self._log_blank()

        header = f'Searching for "{term}" between {start} and {end}'
        self._log_plain(header)
        self._log_plain('Starting search...')
        self._progress_metric_key = 'article_count'
        self._progress_cumulative = 0
        self._progress_year_totals.clear()
        self._processed_dataset_units = 0
        self._processed_years.clear()
        self._first_year_sec_per_unit = None
        self._eta_seconds_remaining = None
        self._progress_total_metric = metric_total_for_dates(start_date_obj, end_date_obj, self._progress_metric_key)
        self.progress.setFormat('%p%')
        if self._progress_total_metric > 0:
            self.progress.setRange(0, self._progress_total_metric)
            self.progress.setValue(0)
            self.progress_details.setText(
                f"0 of {self._progress_total_metric:,} articles (~0.00%) across the selected range "
                "(based on the packaged yearly totals)."
            )
        else:
            self.progress.setRange(0, 0)
            self.progress.setValue(0)
            self.progress_details.setText(
                'Progress will update as results are found (yearly totals unavailable for this range).'
            )
        self.progress_details.show()
        self.progress_eta.hide()
        self.progress_eta.setText('')
        self._start_time = time.time()
        self.logged_years.clear()
        self._cancel_event.clear()
        self._cancel_requested = False
        self._year_timer = self._start_time
        self._set_running_state(True)

        # Launch download in a separate thread
        self.thread = WorkerThread(
            download_data,
            self.parent().project_folder,
            term,
            start,
            end,
            parquet_dir=dataset_folder,
            cancel_event=self._cancel_event,
            cleaning_options={
                'lowercase_articles': self.clean_lowercase_cb.isChecked(),
                'urls_to_pdf': self.clean_urls_cb.isChecked(),
                'collapse_hyphenated_breaks': self.clean_hyphen_cb.isChecked(),
            },
            metadata_enabled=getattr(self.parent(), 'metadata_enabled', True),
            summary_only=summary_only,
        )
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.download_finished)
        self.thread.start()

    def update_progress(self, payload: object):
        now = time.time()
        elapsed = max(0.0, now - (self._start_time or now))

        year = None
        increment = 0
        cumulative = None
        year_total = None
        heartbeat = False
        is_new_year_entry = False
        year_elapsed = None

        if isinstance(payload, dict):
            raw_year = payload.get('year')
            if raw_year:
                year = str(raw_year)
            heartbeat = bool(payload.get('heartbeat'))
            try:
                increment = int(payload.get('increment', 0) or 0)
            except (TypeError, ValueError):
                increment = 0
            year_total_raw = payload.get('year_dataset_total')
            if year_total_raw is not None:
                try:
                    year_total = int(year_total_raw)
                except (TypeError, ValueError):
                    year_total = None
            cumulative_raw = payload.get('cumulative')
            if cumulative_raw is not None:
                try:
                    cumulative = max(0, int(cumulative_raw))
                except (TypeError, ValueError):
                    cumulative = None
        else:
            try:
                increment = int(payload)
            except (TypeError, ValueError):
                increment = 0

        if increment < 0:
            increment = 0

        if cumulative is None:
            cumulative = max(0, self._progress_cumulative + increment)

        self._progress_cumulative = cumulative

        bar_value = cumulative
        display_value = cumulative
        if self._progress_total_metric and self._progress_total_metric > 0:
            if self.progress.maximum() != self._progress_total_metric:
                self.progress.setRange(0, self._progress_total_metric)
            bar_value = min(cumulative, self._progress_total_metric)
            display_value = bar_value
        else:
            if self.progress.minimum() != 0 or self.progress.maximum() != 0:
                self.progress.setRange(0, 0)
            bar_value = 0
            display_value = cumulative
        self.progress.setValue(bar_value)

        if year:
            self._progress_year_totals[year] = self._progress_year_totals.get(year, 0) + increment

        if self._progress_total_metric and self._progress_total_metric > 0:
            pct = (display_value / self._progress_total_metric * 100.0) if self._progress_total_metric else 0.0
            coverage_text = f"{display_value:,} of {self._progress_total_metric:,} articles (~{pct:.2f}%)"
        else:
            coverage_text = f"{display_value:,} articles processed"

        detail_parts = []
        if year:
            if increment > 0:
                if year_total:
                    year_pct = (increment / year_total * 100.0) if year_total else 0.0
                    detail_parts.append(
                        f"{increment:,} matches from {year} (~{year_pct:.2f}% of {int(year_total):,})"
                    )
                else:
                    detail_parts.append(f"{increment:,} matches from {year}")
            elif not heartbeat:
                detail_parts.append(f"No matches from {year}")

        detail_text = coverage_text + (". " + " ".join(detail_parts) if detail_parts else ".")
        self.progress_details.setText(detail_text)
        if not self.progress_details.isVisible():
            self.progress_details.show()

        if year and not heartbeat and year not in self.logged_years:
            year_elapsed = max(0.0, now - (self._year_timer or now))
            self._year_timer = now
            self.logged_years.add(year)
            is_new_year_entry = True
            if increment > 0:
                if year_total:
                    year_pct = (increment / year_total * 100.0) if year_total else 0.0
                    self._log_plain(
                        f"Found {increment:,} articles in {year} (~{year_pct:.2f}% of {int(year_total):,}) "
                        f"— search time {year_elapsed:.1f}s"
                    )
                else:
                    self._log_plain(f"Found {increment:,} articles in {year} — search time {year_elapsed:.1f}s")
            else:
                self._log_plain(f"No matching articles found in {year} — search time {year_elapsed:.1f}s")
        elif not year and increment and not heartbeat:
            self._log_plain(f"Found {increment:,} articles — elapsed {elapsed:.1f}s")

        self._update_time_estimate(year, year_total, year_elapsed, is_new_year_entry)

    def _update_time_estimate(
        self,
        year: Optional[str],
        year_total: Optional[int],
        year_elapsed: Optional[float],
        is_new_year_entry: bool,
    ) -> None:
        if not self._progress_total_metric or self._progress_total_metric <= 0:
            self.progress_eta.hide()
            return

        updated = False
        if is_new_year_entry and year:
            if year_total is None:
                year_total = 0
            if year not in self._processed_years:
                self._processed_years.add(year)
                self._processed_dataset_units += max(0, int(year_total))
                if self._progress_total_metric:
                    self._processed_dataset_units = min(self._processed_dataset_units, self._progress_total_metric)
                if (
                    self._first_year_sec_per_unit is None
                    and year_elapsed is not None
                    and year_elapsed > 0
                    and year_total
                ):
                    self._first_year_sec_per_unit = year_elapsed / max(1, int(year_total))
                updated = True

        remaining_units = max(0, self._progress_total_metric - self._processed_dataset_units)
        if remaining_units == 0 and self.logged_years:
            self._eta_seconds_remaining = 0.0
            self.progress_eta.setText('Estimated time remaining: 0s')
            self.progress_eta.show()
            return

        if self._first_year_sec_per_unit:
            est_seconds = remaining_units * self._first_year_sec_per_unit
            self._eta_seconds_remaining = est_seconds
            self.progress_eta.setText(
                f"Estimated time remaining: {self._format_duration(est_seconds)}"
            )
            self.progress_eta.show()
        else:
            if self._processed_years or updated:
                self.progress_eta.setText('Estimating remaining time…')
                self.progress_eta.show()
            else:
                self.progress_eta.hide()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        if seconds < 60:
            return f"{seconds}s"
        minutes, sec = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {sec}s"
        hours, minutes = divmod(minutes, 60)
        if hours < 24:
            return f"{hours}h {minutes}m"
        days, hours = divmod(hours, 24)
        return f"{days}d {hours}h"

    def _handle_summary_only_output(self, result_payload: Dict[str, Any], elapsed: float) -> Optional[str]:
        per_year = result_payload.get('per_year') or []
        totals = result_payload.get('totals') or {}
        search_term_val = (result_payload.get('search_term') or self._current_term or '').strip()
        raw_term_input = self.search_input.text().strip()
        if not search_term_val and raw_term_input:
            search_term_val = raw_term_input
        csv_term_value = search_term_val if search_term_val else 'All Terms'

        rows: List[Dict[str, Any]] = []
        for entry in per_year:
            year_label = str(entry.get('year', '') or '')
            rows.append({
                'search_term': csv_term_value,
                'year': year_label,
                'keyword_frequency': int(entry.get('keyword_frequency', 0)),
                'total_articles': int(entry.get('article_count', 0)),
                'total_pages': int(entry.get('page_count', 0)),
                'total_issues': int(entry.get('issue_count', 0)),
                'total_newspapers': int(entry.get('newspaper_count', 0)),
                'total_words': int(entry.get('word_count', 0)),
            })

        if not rows:
            self._log_plain('No records found within the selected range.')
            return None

        totals_row = {
            'search_term': csv_term_value,
            'year': 'Total',
            'keyword_frequency': int(totals.get('keyword_frequency', 0)),
            'total_articles': int(totals.get('article_count', 0)),
            'total_pages': int(totals.get('page_count', 0)),
            'total_issues': int(totals.get('issue_count', 0)),
            'total_newspapers': int(totals.get('newspaper_count', 0)),
            'total_words': int(totals.get('word_count', 0)),
        }
        rows.append(totals_row)

        df = pd.DataFrame(rows)
        term_dir = self._current_term_dir
        if not term_dir:
            processed_root = os.path.join(self.parent().project_folder, 'data', 'processed')
            term_dir = os.path.join(processed_root, term_directory_name(search_term_val))
            os.makedirs(term_dir, exist_ok=True)
            self._current_term_dir = term_dir
        else:
            os.makedirs(term_dir, exist_ok=True)

        term_label = search_term_val or 'all'
        start_label = self._current_start or self.start_input.text().strip() or 'start'
        end_label = self._current_end or self.end_input.text().strip() or 'end'
        csv_name = f"{term_label}_{start_label}_{end_label}_yearly_summary.csv"
        csv_path = os.path.join(term_dir, csv_name)

        if os.path.exists(csv_path):
            self._log_plain(f'Overwriting existing yearly summary CSV at {csv_path}')

        try:
            df.to_csv(csv_path, index=False)
        except Exception as exc:
            self._log_plain(f'Yearly summary CSV failed: {exc}')
            return None

        self._log_plain(
            f"Summary-only search finished in {elapsed:.1f}s — "
            f"{totals_row['total_articles']:,} articles, "
            f"{totals_row['total_pages']:,} pages, "
            f"{totals_row['total_issues']:,} issues, "
            f"{totals_row['total_newspapers']:,} newspapers."
        )
        if search_term_val:
            self._log_plain(f"Keyword frequency total: {totals_row['keyword_frequency']:,}")
        self._log_plain(f"Total words: {totals_row['total_words']:,}")
        self._log_link('Yearly summary CSV', csv_path)
        return csv_path

    def _collect_year_counts(self, result_paths: List[str]) -> Tuple[Dict[str, int], Optional[str]]:
        counts: Dict[str, int] = {}
        detected_term: Optional[str] = None
        for path in result_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
            except Exception as exc:
                self._log_plain(f'Yearly summary skipped for {os.path.basename(path)}: {exc}')
                continue
            if not isinstance(payload, dict):
                continue
            term_val = payload.get('search_term')
            if isinstance(term_val, str) and term_val.strip() and not detected_term:
                detected_term = term_val.strip()
            articles = payload.get('articles') or []
            if not isinstance(articles, list):
                continue
            for article in articles:
                if not isinstance(article, dict):
                    continue
                date_val = str(article.get('date') or '')
                year = date_val[:4]
                if year.isdigit():
                    counts[year] = counts.get(year, 0) + 1
        return counts, detected_term

    def cancel_download(self):
        if self._search_running and self.thread and self.thread.isRunning():
            if self._cancel_requested:
                return
            self._cancel_requested = True
            self._cancel_event.set()
            self.cancel_btn.setText('Canceling...')
            self.cancel_btn.setEnabled(False)
            self._log_plain('Cancelling search...')
        else:
            self.close()

    def download_finished(self, result):
        elapsed = time.time() - self._start_time
        self._set_running_state(False)
        if self._progress_total_metric and self._processed_dataset_units >= self._progress_total_metric:
            self.progress_eta.setText('Estimated time remaining: 0s')
            self.progress_eta.show()
        else:
            self.progress_eta.hide()
        summary_only_mode = bool(getattr(self, '_summary_only_requested', False))
        if isinstance(result, Exception):
            QMessageBox.critical(self, 'Error', str(result))
            self._log_plain(f"Search failed: {result}")
            self._cancel_event.clear()
            self._cancel_requested = False
            self._summary_only_requested = False
            self._finalize_project_log()
            return

        if self._cancel_requested:
            if result:
                for path in result:
                    self._log_link('Cancelled after', path, elapsed)
            else:
                self._log_plain(f'Search cancelled after {elapsed:.1f}s — no records saved')
            self._cancel_event.clear()
            self._cancel_requested = False
            self._summary_only_requested = False
            self._finalize_project_log()
            return

        if summary_only_mode and isinstance(result, dict) and result.get('summary_only'):
            self._log_separator()
            self._handle_summary_only_output(result, elapsed)
            self._cancel_event.clear()
            self._cancel_requested = False
            self._summary_only_requested = False
            self._finalize_project_log()
            return

        self._summary_only_requested = False

        self._log_separator()

        # Automatically load the last downloaded JSON
        last_json = result[-1] if result else None
        if last_json:
            p = self.parent()
            p.json_file = last_json
            p._update_loaded_file_labels()
            p.collocation_state = {'dropped_terms': [], 'term_groups': [], 'topic_settings': {}, 'topic_trend_settings': {}}
            p.collocation_drop_terms = []

        total_articles = 0
        total_years = len(self.logged_years)
        for path in result:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    count = payload.get('match_count')
                    if count is None:
                        articles = payload.get('articles')
                        count = len(articles) if isinstance(articles, list) else 0
                    total_articles += int(count)
            except Exception:
                continue

        if result:
            summary = f"Found {total_articles:,} articles across {total_years} year" + ("s" if total_years != 1 else "")
            summary += f" and finished in {elapsed:.1f}s"
            self._log_plain(summary)
            for path in result:
                self._log_link('Saved to', path)
        else:
            summary = f"Found 0 articles across {total_years} year" + ("s" if total_years != 1 else "")
            summary += f" and finished in {elapsed:.1f}s"
            self._log_plain(summary)
            self._log_plain('No JSON created.')

        yearly_counts: Dict[str, int] = {}
        summary_term: Optional[str] = None
        if result and (self.yearly_csv_cb.isChecked() or self.yearly_chart_cb.isChecked()):
            yearly_counts, summary_term = self._collect_year_counts(result)

        if yearly_counts:
            rows: List[Tuple[int, int]] = []
            for year_str, count in yearly_counts.items():
                try:
                    year_int = int(year_str)
                except ValueError:
                    continue
                rows.append((year_int, int(count)))
            rows.sort()

            if rows:
                df_counts = pd.DataFrame(rows, columns=['year', 'article_count'])
                term_label = summary_term or self._current_term or self.search_input.text().strip() or 'term'
                start_label = self._current_start or self.start_input.text().strip() or 'start'
                end_label = self._current_end or self.end_input.text().strip() or 'end'
                df_counts.insert(0, 'search_term', term_label)

                term_dir = self._current_term_dir or (os.path.dirname(result[-1]) if result else None)
                if term_dir:
                    os.makedirs(term_dir, exist_ok=True)

                if self.yearly_csv_cb.isChecked() and term_dir:
                    csv_name = f"{term_label}_{start_label}_{end_label}_yearly_counts.csv"
                    csv_path = os.path.join(term_dir, csv_name)
                    try:
                        df_counts.to_csv(csv_path, index=False)
                    except Exception as exc:
                        self._log_plain(f'Yearly summary CSV failed: {exc}')
                    else:
                        self._log_link('Yearly summary CSV', csv_path)

                if self.yearly_chart_cb.isChecked():
                    try:
                        plot_year = _import_plot_articles_by_year()
                        title = f'Articles per Year — {term_label}'
                        plot_year(df_counts, title=title)
                        self._log_plain('Opened yearly article chart.')
                    except Exception as exc:
                        self._log_plain(f'Yearly chart failed: {exc}')
            else:
                self._log_plain('Yearly summaries skipped — no dated articles found.')
        elif result and (self.yearly_csv_cb.isChecked() or self.yearly_chart_cb.isChecked()):
            self._log_plain('Yearly summaries skipped — no articles with valid dates.')

        geojson_outputs: List[str] = []
        if self.clean_geo_cb.isChecked() and result:
            geojson_outputs = self._add_geo_after_search(result[-1])
            if geojson_outputs:
                total_articles_geo, places_all, _ = _summarize_geojson_outputs(geojson_outputs)
                stats = getattr(geojson_outputs, 'stats', None)
                if stats:
                    total_articles_stat = stats.get('total_articles', total_articles_geo)
                    matched_lccn = stats.get('matched_lccn', 0)
                    matched_title = stats.get('matched_title', 0)
                    matched_total = matched_lccn + matched_title
                    total_base = total_articles_stat or (matched_total + stats.get('unmatched', 0)) or 1
                    self._log_plain(
                        f'Added geographic info for {matched_total:,} articles across {len(places_all):,} locations.'
                    )
                    pct = lambda count: (count / total_base * 100.0) if total_base else 0.0
                    self._log_plain(
                        f"Matched {matched_lccn:,} articles via LCCN ({pct(matched_lccn):.2f}%) and "
                        f"{matched_title:,} via title/date fallback ({pct(matched_title):.2f}%)."
                    )
                    unmatched_total = stats.get('unmatched', max(total_articles_stat - matched_total, 0))
                    self._log_plain(
                        f"{unmatched_total:,} articles had no geographic match ({pct(unmatched_total):.2f}%)."
                    )
                    unmatched_path = stats.get('unmatched_path')
                    if unmatched_path:
                        self._log_link('Unmatched table saved to', unmatched_path)
                    elif self.clean_geo_unmatched_cb.isChecked() and unmatched_total == 0:
                        self._log_plain('No unmatched table created — all articles were matched.')
                else:
                    self._log_plain(
                        f'Added geographic info for {total_articles_geo:,} articles across {len(places_all):,} locations.'
                    )
                for out_path in geojson_outputs:
                    self._log_link('GeoJSON saved to', out_path)
                parent_ref = self.parent()
                if parent_ref is not None:
                    append_geojson_project_log(parent_ref, geojson_outputs)

        self._cancel_event.clear()
        self._cancel_requested = False
        self._finalize_project_log()

class UpdateLocationsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Add Geographic Info')
        self.setMinimumSize(540, 320)

        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._csv_hint_shown = False

        # JSON selection
        self.json_path = getattr(parent, 'json_file', None)
        self.json_label = QLabel(self._display_name(self.json_path))
        json_row = QHBoxLayout()
        json_row.addWidget(self.json_label, 1)
        self.json_change_btn = QPushButton('Change')
        self.json_change_btn.clicked.connect(self.change_json)
        json_row.addWidget(self.json_change_btn)
        form.addRow('Articles JSON:', json_row)

        # CSV selection
        self.csv_path = self._default_csv_path()
        if not (self.csv_path and os.path.exists(self.csv_path)):
            self.csv_path = self.prompt_csv(show_hint=True)
        self._store_selected_csv(self.csv_path)

        self.csv_label = QLabel(self._display_name(self.csv_path))
        csv_row = QHBoxLayout()
        csv_row.addWidget(self.csv_label, 1)
        self.csv_change_btn = QPushButton('Change')
        self.csv_change_btn.clicked.connect(self.change_csv)
        csv_row.addWidget(self.csv_change_btn)
        form.addRow('Locations CSV:', csv_row)

        layout.addLayout(form)

        self.unmatched_checkbox = QCheckBox('Create table of unmatched articles')
        layout.addWidget(self.unmatched_checkbox)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.merge_btn = QPushButton('Create GeoJSON')
        self.merge_btn.clicked.connect(self.perform_merge)
        btn_row.addWidget(self.merge_btn)
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _display_name(self, path: Optional[str]) -> str:
        if not path:
            return '(none)'
        return path if len(path) <= 80 else '…' + path[-77:]

    def change_json(self):
        parent = self.parent()
        start = self.json_path or getattr(parent, 'project_folder', os.getcwd())
        path, _ = QFileDialog.getOpenFileName(self, 'Select Articles JSON', start, 'JSON Files (*.json)')
        if not path:
            return
        self.json_path = path
        parent.json_file = path
        parent._update_loaded_file_labels()
        self.json_label.setText(self._display_name(path))

    def _default_csv_path(self) -> Optional[str]:
        return resolve_locations_csv(self.parent())

    def prompt_csv(self, show_hint: bool = False) -> Optional[str]:
        parent = self.parent()
        if show_hint and not self._csv_hint_shown:
            QMessageBox.information(
                self,
                'Locate Locations CSV',
                f'Locate the newspaper locations table named "{DEFAULT_CSV_FILENAME}".'
            )
            self._csv_hint_shown = True
        start = parent.project_folder if parent else os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            f'Select Locations CSV ({DEFAULT_CSV_FILENAME})',
            start,
            'CSV Files (*.csv)'
        )
        return path

    def change_csv(self):
        path = self.prompt_csv(show_hint=True)
        if not path:
            return
        self.csv_path = path
        self.csv_label.setText(self._display_name(path))
        self._store_selected_csv(path)

    def _store_selected_csv(self, path: Optional[str]):
        if not (path and os.path.exists(path)):
            return
        parent = self.parent()
        if parent is not None:
            parent.locations_csv_path = path

    def perform_merge(self):
        parent = self.parent()
        if not self.json_path or not os.path.exists(self.json_path):
            self.change_json()
            if not self.json_path:
                return

        if not self.csv_path or not os.path.exists(self.csv_path):
            self.change_csv()
            if not self.csv_path:
                return

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            term = None
            year = None
            if isinstance(info, dict):
                term = info.get('search_term')
                year = info.get('year')
            if not term:
                base = os.path.basename(self.json_path)
                term = base.split('_', 1)[0] if base else None
            if not term:
                raise ValueError('Could not infer search_term from JSON or filename.')
        except Exception as exc:
            QMessageBox.critical(self, 'Error', f'Could not read metadata: {exc}')
            return

        unmatched_csv_path = None
        if self.unmatched_checkbox.isChecked():
            base_name = os.path.splitext(os.path.basename(self.json_path))[0]
            unmatched_csv_path = os.path.join(
                parent.project_folder,
                'data',
                'processed',
                f'unmatched_{base_name}.csv'
            )
        try:
            merge_geojson = _import_merge_geojson()
            out_paths = merge_geojson(
                parent.project_folder,
                csv_path=self.csv_path,
                search_term=term,
                year=year,
                json_path=self.json_path,
                unmatched_csv_path=unmatched_csv_path
            )
            if out_paths:
                last_geo = out_paths[-1]
                parent.geojson_file = last_geo
                if parent.locations_csv_path and not os.path.samefile(parent.locations_csv_path, self.csv_path):
                    parent.locations_csv_path = self.csv_path
                parent._update_loaded_file_labels()
            self._log_merge_stats(out_paths)
            self.accept()
            if parent:
                if hasattr(parent, 'raise_'):
                    parent.raise_()
                if hasattr(parent, 'activateWindow'):
                    parent.activateWindow()
        except Exception as exc:
            QMessageBox.critical(self, 'Error', str(exc))

    def _log_merge_stats(self, out_paths):
        parent = self.parent()
        if parent is None:
            return
        append_geojson_project_log(parent, out_paths)

class SortableTableWidgetItem(QTableWidgetItem):
    """Table item that prefers numeric sort using UserRole."""

    def __lt__(self, other: 'SortableTableWidgetItem') -> bool:  # type: ignore[override]
        if isinstance(other, QTableWidgetItem):
            self_val = self.data(Qt.UserRole)
            other_val = other.data(Qt.UserRole)
            if isinstance(self_val, (int, float)) and isinstance(other_val, (int, float)):
                return self_val < other_val
        return super().__lt__(other)


class CSVPreviewDialog(QDialog):
    def __init__(self, csv_path, parent=None, max_rows=100):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(csv_path))
        self.setMinimumSize(900, 600)
        df = pd.read_csv(csv_path).head(max_rows)
        tbl = QTableWidget(df.shape[0], df.shape[1], self)
        tbl.setHorizontalHeaderLabels(list(df.columns))
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        tbl.setSelectionMode(QAbstractItemView.SingleSelection)
        tbl.setAlternatingRowColors(True)

        self._hyperlink_columns = {
            idx for idx, col in enumerate(df.columns) if self._is_hyperlink_column(col)
        }

        topic_columns = {
            idx
            for idx, name in enumerate(df.columns)
            if isinstance(name, str) and name.strip().lower() in {'topic_score', 'topic_weight'}
        }

        for i, row in df.iterrows():
            for j, val in enumerate(row):
                is_topic_metric = j in topic_columns
                display, sort_value = self._render_value(val, numeric=is_topic_metric)
                item = SortableTableWidgetItem(display)
                if sort_value is not None:
                    item.setData(Qt.UserRole, sort_value)
                else:
                    item.setData(Qt.UserRole, None)
                if j in self._hyperlink_columns and display:
                    font = QFont(item.font())
                    font.setUnderline(True)
                    item.setFont(font)
                    item.setForeground(QColor(17, 85, 204))
                    item.setToolTip('Open link in browser')
                tbl.setItem(i, j, item)

        if self._hyperlink_columns:
            tbl.cellActivated.connect(self._handle_link_activation)
        layout = QVBoxLayout(self)
        layout.addWidget(tbl)
        tbl.setFocus()
        tbl.setSortingEnabled(True)
        self.table = tbl
        self._dataframe = df

    @staticmethod
    def _is_hyperlink_column(name: str) -> bool:
        if not name:
            return False
        lowered = str(name).strip().lower()
        tokens = {'url', 'link', 'href'}
        return any(token in lowered for token in tokens)

    def _handle_link_activation(self, row: int, column: int):
        if column not in self._hyperlink_columns:
            return
        item = self.table.item(row, column)
        if item is None:
            return
        raw = item.data(Qt.UserRole)
        if not raw:
            return
        url = str(raw).strip()
        if not url:
            return
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', url):
            url = 'https://' + url
        qurl = QUrl(url)
        if not qurl.isValid():
            return
        QDesktopServices.openUrl(qurl)

    @staticmethod
    def _render_value(value: Any, *, numeric: bool) -> Tuple[str, Optional[float]]:
        if pd.isna(value):
            return '', None
        if not numeric:
            return str(value), None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value), None
        return f"{numeric_value:,.0f}", numeric_value


class CollocationRankSettingsDialog(QDialog):
    def __init__(
        self,
        parent,
        bins: List[str],
        max_terms: int,
        default_top_n: int = 10,
        *,
        selected_terms: Optional[Iterable[str]] = None,
        csv_status: Optional[str] = None,
        csv_path: Optional[str] = None,
        drop_terms: Optional[Iterable[str]] = None,
        log_scale: bool = True,
    ):
        super().__init__(parent)
        self.setWindowTitle('Bump Chart Settings')
        layout = QVBoxLayout(self)

        self._csv_path = csv_path
        self._drop_terms = [str(t).strip() for t in (drop_terms or []) if str(t).strip()]
        self._initial_log_scale = bool(log_scale)
        status_text = ''
        status_kind = (csv_status or '').strip().lower()
        if status_kind == 'created':
            status_text = 'By-time CSV generated for these settings. The first build can take longer than future updates.'
        elif status_kind == 'existing':
            if csv_path and os.path.exists(csv_path):
                folder = os.path.dirname(csv_path) or os.path.abspath(csv_path)
                status_text = (
                    f'Existing by-time CSV located. '
                    f'<a href="open">Open containing folder</a> ({html.escape(folder)}).'
                )
            else:
                status_text = 'Existing by-time CSV located.'
        elif status_kind == 'drop_terms':
            status_text = 'Drop terms active; a new by-time CSV will be generated when you click OK.'
        elif status_kind == 'missing':
            status_text = 'By-time CSV not found yet. It will be generated when you click OK.'
        elif csv_status:
            status_text = html.escape(str(csv_status))

        if status_text:
            info_label = QLabel(status_text)
            info_label.setWordWrap(True)
            info_label.setStyleSheet('color: #555555; font-size: 11px;')
            info_label.setTextFormat(Qt.RichText)
            info_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
            info_label.setOpenExternalLinks(False)
            info_label.linkActivated.connect(self._handle_info_link)
            layout.addWidget(info_label)
        else:
            info_label = None
        self._info_label = info_label

        form = QFormLayout()

        self.top_spin = QSpinBox()
        self.top_spin.setRange(1, max(1, max_terms))
        self.top_spin.setValue(min(default_top_n, max(1, max_terms)))
        form.addRow('Top N terms:', self.top_spin)

        self.home_combo = QComboBox()
        for label in bins:
            self.home_combo.addItem(str(label))
        form.addRow('Home time bin:', self.home_combo)

        self.global_check = QCheckBox('Rank terms across entire time period (ignore home bin)')
        form.addRow(self.global_check)

        self.labels_check = QCheckBox('Show term labels on chart')
        self.labels_check.setChecked(True)
        form.addRow(self.labels_check)

        self.log_scale_check = QCheckBox('Use log scale (y-axis)')
        self.log_scale_check.setChecked(self._initial_log_scale)
        form.addRow(self.log_scale_check)

        layout.addLayout(form)

        self._selected_terms: List[str] = list(dict.fromkeys(selected_terms or []))
        has_selected = bool(self._selected_terms)

        self.use_selected_check = QCheckBox('Use selected terms from Collocation tool (overrides Top N)')
        self.use_selected_check.setChecked(has_selected)
        self.use_selected_check.setEnabled(has_selected)
        layout.addWidget(self.use_selected_check)
        self.use_selected_check.toggled.connect(self._handle_use_selected_toggle)

        self.selection_label = QLabel()
        self.selection_label.setWordWrap(True)
        self.selection_label.setStyleSheet('color: #555555; font-size: 11px;')
        layout.addWidget(self.selection_label)

        self.drop_terms_label = QLabel()
        self.drop_terms_label.setWordWrap(True)
        self.drop_terms_label.setStyleSheet('color: #555555; font-size: 11px;')
        layout.addWidget(self.drop_terms_label)

        self.global_check.toggled.connect(self.home_combo.setDisabled)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_selection_summary()
        self._update_drop_terms_note()

    def values(self) -> dict:
        return {
            'top_n': self.top_spin.value(),
            'home_bin_index': self.home_combo.currentIndex(),
            'use_global': self.global_check.isChecked(),
            'show_labels': self.labels_check.isChecked(),
            'use_selected_terms': self.use_selected_check.isChecked(),
            'selected_terms': list(self._selected_terms),
            'log_scale': self.log_scale_check.isChecked(),
        }

    def _update_selection_summary(self):
        count = len(self._selected_terms)
        if count:
            preview = ', '.join(self._selected_terms[:3])
            if count > 3:
                preview += ', …'
            self.selection_label.setText(
                f'Selected {count} term(s): {html.escape(preview)}.'
            )
        else:
            self.selection_label.setText('No specific terms selected. Top N terms will be used.')
        self._apply_top_spin_enabled()

    def _handle_use_selected_toggle(self, checked: bool):
        if checked and not self._selected_terms:
            self.use_selected_check.setChecked(False)
            return
        self._apply_top_spin_enabled()

    def _apply_top_spin_enabled(self):
        use_selected = self.use_selected_check.isChecked() and bool(self._selected_terms)
        self.top_spin.setEnabled(not use_selected)
        if self._selected_terms:
            base_text = self.selection_label.text().split(' (', 1)[0]
            base_text = base_text.rstrip('.')
            if use_selected:
                self.selection_label.setText(f'{base_text} (overrides Top N).')
            else:
                self.selection_label.setText(f'{base_text}.')

    def _handle_info_link(self, link: str):
        if link != 'open':
            return
        if self._csv_path:
            reveal_in_file_manager(self._csv_path)

    def _update_drop_terms_note(self):
        if not self._drop_terms:
            self.drop_terms_label.setText('No drop terms active; all collocates are considered.')
            return
        preview = ', '.join(self._drop_terms[:3])
        if len(self._drop_terms) > 3:
            preview += ', …'
        self.drop_terms_label.setText(f'Drop terms active ({len(self._drop_terms)}): {html.escape(preview)}')


class TopicTrendsSettingsDialog(QDialog):
    METRIC_OPTIONS = [
        ('Topic Weight (sum)', 'weight_sum'),
        ('Topic Rank (by weight)', 'ordinal_rank'),
        ('Article Count', 'doc_count'),
    ]

    def __init__(self, parent, *, defaults: Optional[dict], max_topics: int):
        super().__init__(parent)
        self.setWindowTitle('Topic Trend Settings')
        layout = QVBoxLayout(self)

        defaults = dict(defaults or {})
        max_topics = max(1, int(max_topics or 1))
        default_top = int(defaults.get('top_topics') or min(12, max_topics))
        default_top = max(1, min(default_top, max_topics))
        default_metric = str(defaults.get('metric') or 'weight_sum')
        default_log = bool(defaults.get('log_scale')) if 'log_scale' in defaults else (default_metric == 'ordinal_rank')
        default_legend = bool(defaults.get('legend_topics', True))
        default_labels = bool(defaults.get('label_points', True))

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft)
        form.setSpacing(6)

        self.topics_spin = QSpinBox()
        self.topics_spin.setRange(1, max_topics)
        self.topics_spin.setValue(default_top)
        form.addRow('Topics to display:', self.topics_spin)

        self.metric_combo = QComboBox()
        for label, value in self.METRIC_OPTIONS:
            self.metric_combo.addItem(label, value)
        metric_index = next((idx for idx, (_, val) in enumerate(self.METRIC_OPTIONS) if val == default_metric), 0)
        self.metric_combo.setCurrentIndex(metric_index)
        form.addRow('Metric to plot:', self.metric_combo)

        self.log_scale_check = QCheckBox('Use log scale')
        self.log_scale_check.setChecked(default_log)
        form.addRow('', self.log_scale_check)

        layout.addLayout(form)

        self.legend_check = QCheckBox('Show legend with topic names')
        self.legend_check.setChecked(default_legend)
        layout.addWidget(self.legend_check)

        self.label_points_check = QCheckBox('Label final points with topic text')
        self.label_points_check.setChecked(default_labels)
        layout.addWidget(self.label_points_check)

        layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._log_scale_user_changed = 'log_scale' in defaults
        self.metric_combo.currentIndexChanged.connect(self._handle_metric_change)
        self.log_scale_check.stateChanged.connect(self._handle_log_scale_changed)
        self._handle_metric_change()

    def _handle_metric_change(self):
        metric = self.metric_combo.currentData()
        if metric == 'ordinal_rank':
            if not self._log_scale_user_changed:
                self.log_scale_check.blockSignals(True)
                self.log_scale_check.setChecked(True)
                self.log_scale_check.blockSignals(False)
        else:
            if not self._log_scale_user_changed:
                self.log_scale_check.blockSignals(True)
                self.log_scale_check.setChecked(False)
                self.log_scale_check.blockSignals(False)

    def _handle_log_scale_changed(self, _state: int):
        self._log_scale_user_changed = True

    def values(self) -> dict:
        return {
            'top_topics': self.topics_spin.value(),
            'metric': self.metric_combo.currentData(),
            'log_scale': self.log_scale_check.isChecked(),
            'legend_topics': self.legend_check.isChecked(),
            'label_points': self.label_points_check.isChecked(),
        }

class CollocateMapSettingsDialog(QDialog):
    def __init__(
        self,
        parent,
        *,
        time_bins: List[Tuple[str, str]],
        cities: List[Tuple[str, str]],
        states: List[str],
        default_top_n: int = 10,
        max_top_n: int = 150,
        selected_terms: Optional[Iterable[str]] = None,
        use_selected_default: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle('Collocate Map Settings')
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.map_type_combo = QComboBox()
        self.map_type_combo.addItem('Select Term Rank Map', 'rank')
        self.map_type_combo.addItem('Top Ranked Term by Location', 'top_term')
        form.addRow('Map type:', self.map_type_combo)

        self.top_spin = QSpinBox()
        self.top_spin.setRange(1, max_top_n)
        self.top_spin.setValue(max(1, min(default_top_n, max_top_n)))
        form.addRow('Top N terms:', self.top_spin)

        self._selected_terms = list(dict.fromkeys(selected_terms or []))
        self.use_selected_terms_check = QCheckBox('Use selected terms from Collocation tool (overrides Top N)')
        has_selected = bool(self._selected_terms)
        default_selected = has_selected and use_selected_default
        self.use_selected_terms_check.setChecked(default_selected)
        self.use_selected_terms_check.setEnabled(has_selected)
        form.addRow(self.use_selected_terms_check)

        self.selected_terms_label = QLabel()
        self.selected_terms_label.setWordWrap(True)
        self.selected_terms_label.setStyleSheet('color: #555555; font-size: 11px;')
        form.addRow(self.selected_terms_label)

        self.colorize_check = QCheckBox('Color markers by collocate article count')
        self.colorize_check.setChecked(True)
        form.addRow(self.colorize_check)

        self.lightweight_check = QCheckBox('Lightweight mode (trim popup payloads)')
        self.lightweight_check.setChecked(True)
        form.addRow(self.lightweight_check)

        self.export_csv_check = QCheckBox('Export collocate CSV with XY')
        self.export_csv_check.setToolTip('Writes a CSV summarizing collocates with coordinates for further analysis.')
        form.addRow(self.export_csv_check)

        time_row = QWidget()
        time_layout = QHBoxLayout(time_row)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setSpacing(6)
        self.enable_time_slider = QCheckBox('Enable time slider')
        self.enable_time_slider.setEnabled(bool(time_bins))
        if time_bins:
            self.enable_time_slider.setChecked(True)
        info_btn = QToolButton()
        info_btn.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        info_btn.setAutoRaise(True)
        info_btn.setStyleSheet('QToolButton { background: transparent; border: none; }')
        info_btn.setToolTip(
            'To change the time bin size, re-run the collocation analysis and adjust the Bin Size and Time Unit options.'
        )
        time_layout.addWidget(self.enable_time_slider)
        time_layout.addWidget(info_btn)
        time_layout.addStretch(1)
        form.addRow('Time controls:', time_row)

        layout.addLayout(form)

        self.use_selected_terms_check.toggled.connect(self._apply_top_spin_enabled)
        self.map_type_combo.currentIndexChanged.connect(lambda *_: self._apply_map_type_constraints())
        self._apply_map_type_constraints()

        scope_box = QGroupBox('Term selection scope')
        scope_layout = QVBoxLayout(scope_box)
        self.global_radio = QRadioButton('Use entire time period')
        self.time_radio = QRadioButton('Home time bin')
        self.global_radio.setChecked(True)
        self.time_radio.setEnabled(bool(time_bins))
        scope_layout.addWidget(self.global_radio)
        scope_layout.addWidget(self.time_radio)
        self.time_combo = QComboBox()
        for key, label in time_bins:
            display = f'{key}: {label}' if label else str(key)
            self.time_combo.addItem(display, key)
        self.time_combo.setEnabled(False)
        scope_layout.addWidget(self.time_combo)
        layout.addWidget(scope_box)

        def _update_time_enabled():
            enabled = self.time_radio.isChecked() and self.time_combo.count() > 0
            self.time_combo.setEnabled(enabled)
            has_bins = self.time_combo.count() > 0
            self.enable_time_slider.setEnabled(has_bins)
            if not has_bins:
                self.enable_time_slider.setChecked(False)

        self.global_radio.toggled.connect(lambda _checked: _update_time_enabled())
        self.time_radio.toggled.connect(lambda _checked: _update_time_enabled())

        location_box = QGroupBox('Location weighting')
        location_layout = QVBoxLayout(location_box)
        self.loc_all_radio = QRadioButton('All cities')
        self.loc_city_radio = QRadioButton('Specific city')
        self.loc_state_radio = QRadioButton('Specific state')
        self.loc_all_radio.setChecked(True)
        location_layout.addWidget(self.loc_all_radio)
        location_layout.addWidget(self.loc_city_radio)
        self.city_combo = QComboBox()
        for city, state in cities:
            label = city or ''
            if state:
                label = f'{label}, {state}' if label else state
            self.city_combo.addItem(label, (city or '', state or ''))
        self.city_combo.setEnabled(False)
        if not cities:
            self.loc_city_radio.setEnabled(False)
        location_layout.addWidget(self.city_combo)
        location_layout.addWidget(self.loc_state_radio)
        self.state_combo = QComboBox()
        for state in states:
            self.state_combo.addItem(state)
        self.state_combo.setEnabled(False)
        if not states:
            self.loc_state_radio.setEnabled(False)
        location_layout.addWidget(self.state_combo)
        layout.addWidget(location_box)

        def _update_location_controls():
            self.city_combo.setEnabled(
                self.loc_city_radio.isEnabled()
                and self.loc_city_radio.isChecked()
                and self.city_combo.count() > 0
            )
            self.state_combo.setEnabled(
                self.loc_state_radio.isEnabled()
                and self.loc_state_radio.isChecked()
                and self.state_combo.count() > 0
            )

        self.loc_all_radio.toggled.connect(lambda _checked: _update_location_controls())
        self.loc_city_radio.toggled.connect(lambda _checked: _update_location_controls())
        self.loc_state_radio.toggled.connect(lambda _checked: _update_location_controls())

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        _update_time_enabled()
        _update_location_controls()

    def _update_selected_terms_summary(self):
        variant = str(self.map_type_combo.currentData() or 'rank').lower()
        if variant == 'top_term':
            self.selected_terms_label.setText('Top ranked term map selects the leading collocate term for each location automatically.')
            return
        count = len(self._selected_terms)
        if count:
            preview = ', '.join(self._selected_terms[:3])
            if count > 3:
                preview += ', …'
            self.selected_terms_label.setText(
                f'Selected {count} term(s): {html.escape(preview)}.'
            )
        else:
            self.selected_terms_label.setText('No selected terms available; Top N terms will be used.')

    def _apply_top_spin_enabled(self):
        variant = str(self.map_type_combo.currentData() or 'rank').lower()
        use_selected = self.use_selected_terms_check.isChecked() and bool(self._selected_terms) and variant != 'top_term'
        self.top_spin.setEnabled(not use_selected)
        if variant == 'top_term':
            return
        if self._selected_terms:
            base_text = self.selected_terms_label.text().split(' (', 1)[0]
            base_text = base_text.rstrip('.')
            if use_selected:
                self.selected_terms_label.setText(f'{base_text} (overrides Top N).')
            else:
                self.selected_terms_label.setText(f'{base_text}.')

    def _apply_map_type_constraints(self):
        variant = str(self.map_type_combo.currentData() or 'rank').lower()
        top_term_mode = variant == 'top_term'
        if top_term_mode:
            self.use_selected_terms_check.setChecked(False)
            self.use_selected_terms_check.setEnabled(False)
            self.use_selected_terms_check.hide()
            self.colorize_check.setChecked(False)
            self.colorize_check.setEnabled(False)
        else:
            self.use_selected_terms_check.setEnabled(bool(self._selected_terms))
            self.use_selected_terms_check.show()
            self.colorize_check.setEnabled(True)
        self._update_selected_terms_summary()
        self._apply_top_spin_enabled()

    def values(self) -> dict:
        term_scope = 'global'
        time_key = None
        time_label = ''
        if self.time_radio.isChecked() and self.time_combo.count() > 0:
            term_scope = 'time'
            data = self.time_combo.currentData()
            time_key = str(data if data is not None else self.time_combo.currentText()).strip()
            display_text = str(self.time_combo.currentText()).strip()
            if display_text:
                if ':' in display_text:
                    time_label = display_text.split(':', 1)[1].strip()
                else:
                    time_label = display_text

        location_scope = 'all'
        city_val = ''
        state_val = ''
        location_label = 'All cities'
        if self.loc_city_radio.isChecked() and self.city_combo.count() > 0:
            data = self.city_combo.currentData()
            if isinstance(data, tuple):
                city_val, state_val = data
            location_scope = 'city'
            location_label = str(self.city_combo.currentText()).strip() or 'Selected city'
        elif self.loc_state_radio.isChecked() and self.state_combo.count() > 0:
            location_scope = 'state'
            state_val = str(self.state_combo.currentText()).strip()
            location_label = f'State: {state_val}' if state_val else 'Selected state'

        return {
            'map_type': self.map_type_combo.currentData(),
            'top_n': self.top_spin.value(),
            'term_scope': term_scope,
            'time_key': time_key,
            'time_label': time_label,
            'location_scope': location_scope,
            'location_city': str(city_val or ''),
            'location_state': str(state_val or ''),
            'colorize': self.colorize_check.isChecked(),
            'lightweight': self.lightweight_check.isChecked(),
            'location_label': location_label,
            'enable_time_slider': self.enable_time_slider.isEnabled() and self.enable_time_slider.isChecked(),
            'use_selected_terms': self.use_selected_terms_check.isChecked(),
            'export_csv': self.export_csv_check.isChecked(),
        }


class MapToolDialog(QDialog):
    METRIC_OPTIONS = [
        {
            'label': 'Articles',
            'value': 'article_count',
            'denominator': 'article_count',
            'metric_display': 'Articles',
            'normalized_display': 'Articles / Total Articles',
            'denom_label': 'total articles',
        },
        {
            'label': 'Page count',
            'value': 'page_count',
            'denominator': 'page_count',
            'metric_display': 'Pages',
            'normalized_display': 'Pages / Total Pages',
            'denom_label': 'total pages',
        },
        {
            'label': 'Term frequency',
            'value': 'key_term_frequency',
            'denominator': 'word_count',
            'metric_display': 'Term Frequency',
            'normalized_display': 'Term Frequency / Total Words',
            'denom_label': 'total words',
        },
    ]

    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)
        self.setWindowTitle('Create Map')
        self.setMinimumSize(560, 560)
        self._parent = parent
        defaults = defaults or getattr(parent, 'map_settings', {})
        self._metric_map = {opt['value']: opt for opt in self.METRIC_OPTIONS}
        self.geojson_path = getattr(parent, 'geojson_file', None)

        main_layout = QVBoxLayout(self)

        form = QFormLayout()

        geo_row = QWidget()
        geo_layout = QHBoxLayout(geo_row)
        geo_layout.setContentsMargins(0, 0, 0, 0)
        self.geojson_display = QLabel()
        self.geojson_display.setWordWrap(True)
        geo_layout.addWidget(self.geojson_display, 1)
        self.browse_geojson_btn = QPushButton('Browse…')
        self.browse_geojson_btn.clicked.connect(self._choose_geojson)
        geo_layout.addWidget(self.browse_geojson_btn, 0)
        form.addRow('GeoJSON:', geo_row)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem('Point Map', 'points')
        self.mode_combo.addItem('Graduated Symbols', 'graduated')
        self.mode_combo.addItem('Heat Map', 'heatmap')
        self.mode_combo.addItem('Cluster Map', 'cluster')
        form.addRow('Map type:', self.mode_combo)

        self.metric_combo = QComboBox()
        for opt in self.METRIC_OPTIONS:
            self.metric_combo.addItem(opt['label'], opt['value'])
        form.addRow('Metric:', self.metric_combo)

        self.normalize_check = QCheckBox()
        form.addRow('Normalization:', self.normalize_check)

        units = ['day', 'week', 'month', 'year']

        self.disable_time = QCheckBox('Disable time animation')
        form.addRow('', self.disable_time)

        time_row = QWidget()
        time_layout = QHBoxLayout(time_row)
        time_layout.setContentsMargins(0, 0, 0, 0)
        self.time_step = QSpinBox()
        self.time_step.setRange(1, 104)
        self.time_step.setMaximumWidth(80)
        time_layout.addWidget(self.time_step)
        self.time_unit = QComboBox()
        self.time_unit.addItems(units)
        time_layout.addWidget(self.time_unit)
        form.addRow('Time bin:', time_row)
        self._time_row = time_row

        linger_row = QWidget()
        linger_layout = QHBoxLayout(linger_row)
        linger_layout.setContentsMargins(0, 0, 0, 0)
        self.linger_step = QSpinBox()
        self.linger_step.setRange(0, 104)
        self.linger_step.setMaximumWidth(80)
        linger_layout.addWidget(self.linger_step)
        self.linger_unit = QComboBox()
        self.linger_unit.addItems(units)
        linger_layout.addWidget(self.linger_unit)
        form.addRow('Linger:', linger_row)
        self._linger_row = linger_row

        heat_row = QWidget()
        heat_layout = QHBoxLayout(heat_row)
        heat_layout.setContentsMargins(0, 0, 0, 0)
        self.heat_radius = QSpinBox()
        self.heat_radius.setRange(1, 160)
        heat_layout.addWidget(self.heat_radius)
        form.addRow('Heat radius:', heat_row)
        self._heat_row = heat_row

        heat_value_row = QWidget()
        heat_value_layout = QHBoxLayout(heat_value_row)
        heat_value_layout.setContentsMargins(0, 0, 0, 0)
        self.heat_value = QDoubleSpinBox()
        self.heat_value.setRange(0.1, 50.0)
        self.heat_value.setSingleStep(0.1)
        self.heat_value.setDecimals(2)
        heat_value_layout.addWidget(self.heat_value)
        form.addRow('Heat value:', heat_value_row)
        self._heat_value_row = heat_value_row

        grad_row = QWidget()
        grad_layout = QHBoxLayout(grad_row)
        grad_layout.setContentsMargins(0, 0, 0, 0)
        grad_layout.addWidget(QLabel('Min:'))
        self.grad_min_radius = QSpinBox()
        self.grad_min_radius.setRange(1, 200)
        grad_layout.addWidget(self.grad_min_radius)
        grad_layout.addSpacing(10)
        grad_layout.addWidget(QLabel('Max:'))
        self.grad_max_radius = QSpinBox()
        self.grad_max_radius.setRange(1, 240)
        grad_layout.addWidget(self.grad_max_radius)
        form.addRow('Graduated radii:', grad_row)
        self._grad_row = grad_row

        self.lightweight_check = QCheckBox('Lightweight output (trim popups and tables)')
        form.addRow('Lightweight:', self.lightweight_check)

        self.table_mode_combo = QComboBox()
        self.table_mode_combo.addItem('Full Table', 'full')
        self.table_mode_combo.addItem('Article Only', 'article')
        self.table_mode_combo.addItem('Minimal', 'minimal')
        form.addRow('Attribute table:', self.table_mode_combo)

        self.table_row_limit = QSpinBox()
        self.table_row_limit.setRange(0, 1_000_000)
        self.table_row_limit.setSpecialValueText('All rows')
        self.table_row_limit.setMaximumWidth(120)
        form.addRow('Table row limit:', self.table_row_limit)

        main_layout.addLayout(form)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet('color: #2b6cb0;')
        main_layout.addWidget(self.status_label)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.create_btn = QPushButton('Create Map')
        self.create_btn.clicked.connect(self._run_create_map)
        button_row.addWidget(self.create_btn)
        self.close_btn = QPushButton('Close')
        self.close_btn.clicked.connect(self.close)
        button_row.addWidget(self.close_btn)
        main_layout.addLayout(button_row)

        self._user_row_limit_override = False
        self._auto_row_limit = False
        self._current_table_mode = None
        self._apply_defaults(defaults)
        self._update_geojson_label()
        self.mode_combo.currentIndexChanged.connect(self._update_enabled_state)
        self.disable_time.toggled.connect(self._update_enabled_state)
        self.normalize_check.toggled.connect(self._update_enabled_state)
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        self.grad_min_radius.valueChanged.connect(self._sync_grad_radii)
        self.grad_max_radius.valueChanged.connect(self._sync_grad_radii)
        self.table_mode_combo.currentIndexChanged.connect(self._on_table_mode_changed)
        self.table_row_limit.valueChanged.connect(self._on_row_limit_changed)
        self._update_enabled_state()

    def _apply_defaults(self, defaults: dict):
        mode_def = str(defaults.get('mode', 'points')).lower()
        idx = self.mode_combo.findData(mode_def)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

        metric_def = str(defaults.get('metric', 'article_count'))
        idx = self.metric_combo.findData(metric_def)
        if idx >= 0:
            self.metric_combo.setCurrentIndex(idx)

        self.normalize_check.setChecked(bool(defaults.get('normalize', False)))

        self.time_step.setValue(max(1, int(defaults.get('time_step', 1))))
        idx = self.time_unit.findText(str(defaults.get('time_unit', 'month')), Qt.MatchFixedString)
        if idx >= 0:
            self.time_unit.setCurrentIndex(idx)

        self.disable_time.setChecked(bool(defaults.get('disable_time', False)))

        self.linger_step.setValue(max(0, int(defaults.get('linger_step', 2))))
        idx = self.linger_unit.findText(str(defaults.get('linger_unit', 'month')), Qt.MatchFixedString)
        if idx >= 0:
            self.linger_unit.setCurrentIndex(idx)

        self.heat_radius.setValue(max(1, int(defaults.get('heat_radius', 15))))
        self.heat_value.setValue(float(defaults.get('heat_value', 1.0)))

        self.grad_min_radius.setValue(max(1, int(defaults.get('grad_min_radius', 6))))
        self.grad_max_radius.setValue(max(self.grad_min_radius.value() + 1, int(defaults.get('grad_max_radius', 28))))

        self.lightweight_check.setChecked(bool(defaults.get('lightweight', False)))
        self._update_normalize_text()

        table_mode_def = str(defaults.get('table_mode', 'full')).lower()
        idx = self.table_mode_combo.findData(table_mode_def)
        if idx >= 0:
            self.table_mode_combo.setCurrentIndex(idx)

        row_limit_def = int(defaults.get('table_row_limit', 0) or 0)
        self.table_row_limit.blockSignals(True)
        self.table_row_limit.setValue(max(0, row_limit_def))
        self.table_row_limit.blockSignals(False)
        self._user_row_limit_override = row_limit_def > 0
        self._on_table_mode_changed()

    def _metric_info(self) -> dict:
        key = self.metric_combo.currentData()
        return self._metric_map.get(key, self.METRIC_OPTIONS[0])

    def _update_normalize_text(self):
        info = self._metric_info()
        denom_text = info.get('denom_label', '')
        if denom_text:
            text = f'Normalize by {denom_text} (per city)'
        else:
            text = 'Normalize'
        self.normalize_check.setText(text)

    def _on_metric_changed(self):
        self._update_normalize_text()
        self._update_enabled_state()

    def _sync_grad_radii(self):
        if self.grad_max_radius.value() <= self.grad_min_radius.value():
            self.grad_max_radius.blockSignals(True)
            self.grad_max_radius.setValue(self.grad_min_radius.value() + 1)
            self.grad_max_radius.blockSignals(False)

    def _update_geojson_label(self):
        if self.geojson_path and os.path.exists(self.geojson_path):
            self.geojson_display.setText(html.escape(self.geojson_path))
        else:
            self.geojson_display.setText('<span style="color:#666;">No GeoJSON selected</span>')

    def _choose_geojson(self):
        start_dir = os.path.dirname(self.geojson_path) if self.geojson_path else getattr(self._parent, 'project_folder', os.getcwd())
        path, _ = QFileDialog.getOpenFileName(self, 'Select GeoJSON File', start_dir, 'GeoJSON Files (*.geojson *.json)')
        if path:
            self.geojson_path = path
            if self._parent:
                self._parent.geojson_file = path
                self._parent._update_loaded_file_labels()
            self._update_geojson_label()

    def _update_enabled_state(self):
        mode = self.mode_combo.currentData()
        heat_mode = mode == 'heatmap'
        grad_mode = mode == 'graduated'

        time_controls_enabled = heat_mode and not self.disable_time.isChecked()

        self.disable_time.setEnabled(heat_mode)
        self._time_row.setVisible(heat_mode)
        self._linger_row.setVisible(heat_mode)
        for widget in (self.time_step, self.time_unit, self.linger_step, self.linger_unit):
            widget.setEnabled(time_controls_enabled)

        self._heat_row.setVisible(heat_mode)
        self._heat_value_row.setVisible(heat_mode)
        self._grad_row.setVisible(grad_mode)

        self.normalize_check.setEnabled(True)
        self._update_normalize_text()

    def _on_table_mode_changed(self, *args):
        prev_mode = getattr(self, '_current_table_mode', None)
        mode = (self.table_mode_combo.currentData() or 'full').lower()
        if mode == 'minimal':
            if not self._user_row_limit_override and self.table_row_limit.value() == 0:
                self.table_row_limit.blockSignals(True)
                self.table_row_limit.setValue(1000)
                self.table_row_limit.blockSignals(False)
                self._auto_row_limit = True
                self._user_row_limit_override = False
        else:
            if prev_mode == 'minimal':
                self.table_row_limit.blockSignals(True)
                self.table_row_limit.setValue(0)
                self.table_row_limit.blockSignals(False)
                self._auto_row_limit = False
                self._user_row_limit_override = False
            elif self._auto_row_limit and not self._user_row_limit_override and self.table_row_limit.value() != 0:
                self.table_row_limit.blockSignals(True)
                self.table_row_limit.setValue(0)
                self.table_row_limit.blockSignals(False)
                self._auto_row_limit = False
                self._user_row_limit_override = False
        self._current_table_mode = mode

    def _on_row_limit_changed(self, value: int):
        self._user_row_limit_override = value > 0
        self._auto_row_limit = False

    def _collect_config(self) -> dict:
        info = self._metric_info()
        cfg = {
            'mode': self.mode_combo.currentData(),
            'time_unit': self.time_unit.currentText(),
            'time_step': self.time_step.value(),
            'linger_unit': self.linger_unit.currentText(),
            'linger_step': self.linger_step.value(),
            'disable_time': self.disable_time.isChecked(),
            'heat_radius': self.heat_radius.value(),
            'heat_value': self.heat_value.value(),
            'grad_min_radius': self.grad_min_radius.value(),
            'grad_max_radius': self.grad_max_radius.value(),
            'metric': info['value'],
            'normalize': self.normalize_check.isChecked(),
            'normalize_denominator': info['denominator'] if self.normalize_check.isChecked() else None,
            'lightweight': self.lightweight_check.isChecked(),
            'table_mode': self.table_mode_combo.currentData(),
            'table_row_limit': self.table_row_limit.value() if self.table_row_limit.value() > 0 else 0,
        }
        return cfg

    def _run_create_map(self):
        if not self.geojson_path or not os.path.exists(self.geojson_path):
            QMessageBox.warning(self, 'GeoJSON Required', 'Please select a GeoJSON file to map.')
            return

        cfg = self._collect_config()
        parent = self._parent
        if parent is not None:
            parent.map_settings = dict(cfg)
            parent.geojson_file = self.geojson_path
            parent._update_loaded_file_labels()

        self.create_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.BusyCursor)
        try:
            result = create_map(
                self.geojson_path,
                mode=cfg['mode'],
                time_unit=cfg['time_unit'],
                time_step=cfg['time_step'],
                linger_unit=cfg['linger_unit'],
                linger_step=cfg['linger_step'],
                disable_time=cfg['disable_time'],
                heat_radius=cfg.get('heat_radius'),
                heat_value=cfg.get('heat_value'),
                grad_min_radius=cfg.get('grad_min_radius'),
                grad_max_radius=cfg.get('grad_max_radius'),
                metric=cfg.get('metric'),
                normalize=cfg.get('normalize'),
                normalize_denominator=cfg.get('normalize_denominator'),
                lightweight=cfg.get('lightweight'),
                table_mode=cfg.get('table_mode'),
                table_row_limit=cfg.get('table_row_limit'),
                collocate_term_groups=getattr(parent, 'collocation_term_groups', []),
                metadata_enabled=getattr(parent, 'metadata_enabled', True) if parent else True,
                project_dir=parent.project_folder if parent else None,
            )
        except Exception as exc:
            QMessageBox.critical(self, 'Map Error', f'Failed to create map:\n{exc}')
            result = None
        finally:
            QApplication.restoreOverrideCursor()
            self.create_btn.setEnabled(True)

        if not result:
            return

        map_path = result.get('map_path') if isinstance(result, dict) else result
        if not map_path:
            QMessageBox.critical(self, 'Map Error', 'Map creation did not return an output path.')
            return

        summary = result.get('summary', {}) if isinstance(result, dict) else {}
        self._display_status(summary)

        import webbrowser
        webbrowser.open('file://' + os.path.abspath(map_path))

        if parent is not None:
            attr_path = result.get('attribute_table') if isinstance(result, dict) else None
            parent.append_project_log('Create Map', self._build_log_lines(cfg, map_path, attr_path, summary))

    def _build_log_lines(self, cfg: dict, map_path: str, attr_path: Optional[str], summary: dict) -> list:
        lines = []
        geojson_link = self._link_html(self.geojson_path, 'GeoJSON file') if self.geojson_path else 'Unknown'
        lines.append(f'<div><strong>GeoJSON:</strong> {geojson_link}</div>')
        lines.append(f'<div><strong>Mode:</strong> {html.escape(cfg["mode"])}</div>')
        metric_info = self._metric_map.get(cfg.get('metric'), self.METRIC_OPTIONS[0])
        metric_label = metric_info.get('label', cfg.get('metric', ''))
        lines.append(f'<div><strong>Metric:</strong> {html.escape(metric_label)}</div>')
        if cfg.get('normalize'):
            norm_text = metric_info.get('normalized_display', 'Normalized')
            lines.append(f'<div><strong>Normalization:</strong> {html.escape(norm_text)}</div>')
        lines.append(f'<div><strong>Lightweight:</strong> {"Yes" if cfg.get("lightweight") else "No"}</div>')
        table_mode_value = str(cfg.get('table_mode') or 'full')
        table_mode_label = {
            'full': 'Full Table',
            'article': 'Article Only',
            'minimal': 'Minimal',
        }.get(table_mode_value, table_mode_value.title())
        row_limit_val = cfg.get('table_row_limit') or 0
        row_limit_text = 'All rows' if not row_limit_val else f'{row_limit_val:,} rows'
        lines.append(f'<div><strong>Attribute table:</strong> {html.escape(table_mode_label)} ({row_limit_text})</div>')
        if cfg.get('mode') == 'heatmap' and not cfg.get('disable_time'):
            lines.append(f'<div><strong>Time bin:</strong> {cfg["time_step"]} {html.escape(cfg["time_unit"])}</div>')
            lines.append(f'<div><strong>Linger:</strong> {cfg["linger_step"]} {html.escape(cfg["linger_unit"])}</div>')
        map_link = self._link_html(map_path, 'Open map file')
        lines.append(f'<div><strong>Map output:</strong> {map_link}</div>')

        if attr_path:
            lines.append(f'<div><strong>Attribute table:</strong> {self._link_html(attr_path, "Open attribute table")}</div>')
        if summary:
            summary_parts = []
            if summary.get('term'):
                summary_parts.append(f"Term: {html.escape(summary['term'])}")
            if summary.get('date_range'):
                summary_parts.append(f"Dates: {html.escape(' – '.join(summary['date_range']))}")
            summary_parts.append(f"Articles: {summary.get('articles', 'n/a')}")
            summary_parts.append(f"Newspapers: {summary.get('newspapers', 'n/a')}")
            summary_parts.append(f"Cities: {summary.get('cities', 'n/a')}")
            if summary.get('metric_display'):
                summary_parts.append(f"Mapped metric: {html.escape(summary['metric_display'])}")
            summary_text = '; '.join(summary_parts)
            lines.append(f'<div><strong>Summary:</strong> {summary_text}</div>')
        return lines

    @staticmethod
    def _link_html(path: Optional[str], label: str) -> str:
        if not path:
            return html.escape(label)
        encoded = urllib.parse.quote(path)
        return f'{html.escape(path)} [<a href="chronam-open:{encoded}">Open in Finder</a>]'

    def _display_status(self, summary: dict):
        if not summary:
            self.status_label.setText('Map created successfully.')
            return
        parts = []
        if summary.get('term'):
            parts.append(f"Term: {summary['term']}")
        if summary.get('articles') is not None:
            parts.append(f"Articles: {summary['articles']}")
        if summary.get('newspapers') is not None and summary.get('cities') is not None:
            parts.append(f"Newspapers: {summary['newspapers']}, Cities: {summary['cities']}")
        if summary.get('metric_display'):
            parts.append(f"Metric: {summary['metric_display']}")
        table_mode = summary.get('table_mode')
        if table_mode:
            label = {
                'full': 'Full Table',
                'article': 'Article Only',
                'minimal': 'Minimal',
            }.get(str(table_mode), str(table_mode))
            row_limit = summary.get('table_row_limit') or 0
            limit_text = 'All rows' if not row_limit else f'{row_limit:,} rows'
            parts.append(f"Table: {label} ({limit_text})")
        self.status_label.setText(html.escape('Map created successfully. ' + '; '.join(parts)))


class TermSelectionDialog(QDialog):
    def __init__(
        self,
        parent: Optional[QWidget],
        terms: List[dict],
        selected_terms: Iterable[str],
        *,
        window_title: str,
        info_text: str,
        action_verb: str,
    ):
        super().__init__(parent)
        self.setWindowTitle(window_title)
        self.setMinimumSize(480, 620)
        self.selected_terms: List[str] = list(selected_terms)
        self._initializing = True
        self._action_verb = action_verb
        self._term_frequency: Dict[str, Optional[float]] = {}
        self._item_by_term: Dict[str, QListWidgetItem] = {}
        self._item_original_text: Dict[str, str] = {}

        layout = QVBoxLayout(self)
        self._main_layout = layout
        info = QLabel(info_text)
        info.setWordWrap(True)
        layout.addWidget(info)
        self.info_label = info

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText('Search terms...')
        layout.addWidget(self.search_box)

        length_row = QHBoxLayout()
        length_label = QLabel('Select terms shorter than:')
        self.length_spin = QSpinBox()
        self.length_spin.setRange(0, 50)
        self.length_spin.setSpecialValueText('Off')
        length_row.addWidget(length_label)
        length_row.addWidget(self.length_spin)
        length_row.addStretch(1)
        layout.addLayout(length_row)

        controls = QHBoxLayout()
        self.controls_layout = controls
        self.show_selected_btn = QPushButton('Show Selected Terms')
        self.show_selected_btn.setCheckable(True)
        controls.addWidget(self.show_selected_btn)
        self.clear_btn = QPushButton('Clear Selection')
        controls.addWidget(self.clear_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        layout.addWidget(self.list_widget, 1)

        existing = set()
        self.list_widget.blockSignals(True)
        for info_row in terms:
            term = str(info_row.get('term', '')).strip()
            if not term or term in existing:
                continue
            existing.add(term)
            rank = info_row.get('rank')
            frequency = info_row.get('frequency')
            try:
                freq_numeric = float(frequency)
            except (TypeError, ValueError):
                freq_numeric = None
            self._term_frequency[term] = freq_numeric
            parts = []
            if isinstance(rank, int):
                parts.append(f"#{rank}")
            parts.append(term)
            if frequency is not None:
                freq_text = f"{frequency:g}" if isinstance(frequency, (float, int)) else str(frequency)
                parts.append(f"({freq_text})")
            if rank is None:
                parts.append('(not in current metrics)')
            item_text = ' '.join(parts)
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, term)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if term in self.selected_terms else Qt.Unchecked)
            self.list_widget.addItem(item)
            self._item_by_term[term] = item
            self._item_original_text[term] = item_text
        self.list_widget.blockSignals(False)

        button_row = QHBoxLayout()
        self._button_row_layout = button_row
        button_row.addStretch(1)
        self.action_btn = QPushButton(f'{self._action_verb} 0 Term(s)')
        self.action_btn.setDefault(True)
        self.action_btn.setAutoDefault(True)
        self.cancel_btn = QPushButton('Cancel')
        button_row.addWidget(self.action_btn)
        button_row.addWidget(self.cancel_btn)
        layout.addLayout(button_row)

        self.search_box.textChanged.connect(self._apply_filters)
        self.show_selected_btn.toggled.connect(self._on_toggle_show_selected)
        self.clear_btn.clicked.connect(self._clear_checks)
        self.length_spin.valueChanged.connect(self._handle_length_selection)
        self.list_widget.itemChanged.connect(self._handle_item_changed)
        self.action_btn.clicked.connect(self._accept_selection)
        self.cancel_btn.clicked.connect(self.reject)

        self._initializing = False

        if self.selected_terms:
            self.show_selected_btn.setChecked(True)
        else:
            self._update_show_selected_button_text(self.show_selected_btn.isChecked())

        self._apply_filters()
        self._update_action_button_text()

    def _gather_selected(self) -> List[str]:
        terms: List[str] = []
        seen = set()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                term = item.data(Qt.UserRole)
                if term and term not in seen:
                    seen.add(term)
                    terms.append(term)
        return terms

    def _apply_filters(self):
        search = self.search_box.text().strip().lower()
        show_selected = self.show_selected_btn.isChecked()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            term = item.data(Qt.UserRole) or ''
            matches_search = search in term.lower()
            matches_selection = (not show_selected) or item.checkState() == Qt.Checked
            item.setHidden(not (matches_search and matches_selection))
        self._update_show_selected_button_text(show_selected)
        self._update_action_button_text()

    def _clear_checks(self):
        self.list_widget.blockSignals(True)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Unchecked)
        self.list_widget.blockSignals(False)
        self.length_spin.blockSignals(True)
        self.length_spin.setValue(0)
        self.length_spin.blockSignals(False)
        self.show_selected_btn.setChecked(False)
        self._apply_filters()


    def _handle_length_selection(self, value: int):
        if self._initializing or value <= 0:
            return
        self.list_widget.blockSignals(True)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            term = item.data(Qt.UserRole) or ''
            if term and len(term) <= value:
                item.setCheckState(Qt.Checked)
        self.list_widget.blockSignals(False)
        if not self.show_selected_btn.isChecked():
            self.show_selected_btn.setChecked(True)
        else:
            self._apply_filters()

    def _handle_item_changed(self, _item):
        if self.show_selected_btn.isChecked():
            self._apply_filters()
        else:
            self._update_action_button_text()

    def _accept_selection(self):
        self.selected_terms = self._gather_selected()
        self.accept()

    def _update_action_button_text(self):
        count = len(self._gather_selected())
        self.action_btn.setText(f'{self._action_verb} {count} Term(s)')
        self.action_btn.setEnabled(count > 0)

    def _on_toggle_show_selected(self, checked: bool):
        self._update_show_selected_button_text(checked)
        self._apply_filters()

    def _update_show_selected_button_text(self, checked: bool):
        self.show_selected_btn.setText('Show All Terms' if checked else 'Show Selected Terms')


class TermDropDialog(TermSelectionDialog):
    def __init__(self, parent: Optional[QWidget], terms: List[dict], selected_terms: Iterable[str]):
        super().__init__(
            parent,
            terms,
            selected_terms,
            window_title='Select Terms to Drop',
            info_text='Select collocate terms to exclude from the analysis (top 150 shown).',
            action_verb='Drop',
        )


class TermPlotDialog(TermSelectionDialog):
    def __init__(self, parent: Optional[QWidget], terms: List[dict], selected_terms: Iterable[str]):
        super().__init__(
            parent,
            terms,
            selected_terms,
            window_title='Select Terms to Visualize',
            info_text='Select specific collocate terms to use in charts and maps (top 150 shown).',
            action_verb='Select',
        )


class TermGroupDialog(TermSelectionDialog):
    def __init__(
        self,
        parent: Optional[QWidget],
        terms: List[dict],
        existing_groups: Iterable[dict],
    ):
        self.groups: List[dict] = []
        self.created_groups: List[dict] = []
        self._group_term_lookup: Dict[str, str] = {}
        self._group_name_set: Set[str] = set()
        super().__init__(
            parent,
            terms,
            [],
            window_title='Group Terms',
            info_text='Select collocate terms to group together, then assign a display name for the group.',
            action_verb='Create',
        )

        self.group_btn = QPushButton('Group Terms')
        self.group_btn.clicked.connect(self._create_group_from_selection)
        self.controls_layout.insertWidget(0, self.group_btn)

        groups_box = QGroupBox('Current Groups')
        groups_layout = QVBoxLayout(groups_box)
        groups_layout.setContentsMargins(8, 8, 8, 8)
        self.groups_list = QListWidget()
        self.groups_list.setSelectionMode(QListWidget.SingleSelection)
        groups_layout.addWidget(self.groups_list)

        group_buttons = QHBoxLayout()
        self.edit_group_btn = QPushButton('Edit')
        group_buttons.addWidget(self.edit_group_btn)
        self.rename_group_btn = QPushButton('Rename')
        self.remove_group_btn = QPushButton('Remove')
        group_buttons.addWidget(self.rename_group_btn)
        group_buttons.addWidget(self.remove_group_btn)
        group_buttons.addStretch(1)
        groups_layout.addLayout(group_buttons)

        # Explicit naming option
        self.explicit_names_box = QCheckBox('Explicit Group Names')
        self.explicit_names_box.setToolTip('If checked, new group names default to "*<term1> (<term1>; <term2>; …)". You can still rename.')
        groups_layout.addWidget(self.explicit_names_box)

        insert_index = max(0, self._main_layout.count() - 1)
        self._main_layout.insertWidget(insert_index, groups_box)

        self.groups_list.itemSelectionChanged.connect(self._update_group_controls)
        self.groups_list.itemDoubleClicked.connect(lambda _item: self._rename_selected_group())
        self.edit_group_btn.clicked.connect(self._edit_selected_group)
        self.rename_group_btn.clicked.connect(self._rename_selected_group)
        self.remove_group_btn.clicked.connect(self._remove_selected_group)

        self._load_existing_groups(existing_groups)
        self._update_group_controls()
        self._update_group_button_enabled()
        self._update_action_button_text()

    def _load_existing_groups(self, existing_groups: Iterable[dict]):
        for entry in existing_groups or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get('name', '')).strip()
            if not name:
                continue
            terms_raw = entry.get('terms') or []
            terms: List[str] = []
            seen: Set[str] = set()
            for term in terms_raw:
                term_str = str(term).strip()
                if not term_str:
                    continue
                lower = term_str.lower()
                if lower in seen:
                    continue
                seen.add(lower)
                terms.append(term_str)
            total_freq = entry.get('total_frequency')
            freq_candidates = [self._term_frequency.get(term) for term in terms if self._term_frequency.get(term) is not None]
            if freq_candidates:
                total_freq = float(sum(freq_candidates))
            missing_terms = [term for term in terms if term not in self._item_by_term]
            group_data = {
                'name': name,
                'terms': terms,
            }
            if total_freq is not None:
                group_data['total_frequency'] = total_freq
            if missing_terms:
                group_data['missing_terms'] = list(missing_terms)
            self.groups.append(group_data)
            self._group_name_set.add(name.lower())
            for term in terms:
                self._group_term_lookup[term.lower()] = name
                if term in self._item_by_term:
                    self._set_item_group_state(term, name)
        self._refresh_group_list()

    def _selected_terms_available(self) -> bool:
        for term in self._gather_selected():
            if term.lower() not in self._group_term_lookup:
                return True
        return False

    def _create_group_from_selection(self):
        selected_terms = [term for term in self._gather_selected() if term.lower() not in self._group_term_lookup]
        if not selected_terms:
            QMessageBox.information(self, 'Select Terms', 'Select one or more ungrouped terms to create a new group.')
            return
        # Default name with leading marker
        term1 = selected_terms[0]
        if self.explicit_names_box.isChecked():
            parts = '; '.join(selected_terms)
            default_name = f"*{term1} ({parts})"
        else:
            default_name = f"*{term1}"
        name, ok = QInputDialog.getText(self, 'Group Name', 'Enter display name for this group:', QLineEdit.Normal, default_name)
        if not ok:
            return
        display_name = str(name).strip()
        if not display_name:
            QMessageBox.warning(self, 'Name Required', 'Enter a non-empty display name for the group.')
            return
        if display_name.lower() in self._group_name_set:
            QMessageBox.warning(self, 'Duplicate Name', 'A group with this name already exists. Choose a different name.')
            return
        terms_sorted = list(dict.fromkeys(selected_terms))
        freq_values = [self._term_frequency.get(term) for term in terms_sorted if self._term_frequency.get(term) is not None]
        total_freq = float(sum(freq_values)) if freq_values else None
        group_data: Dict[str, Any] = {
            'name': display_name,
            'terms': terms_sorted,
        }
        if total_freq is not None:
            group_data['total_frequency'] = total_freq
        self.groups.append(group_data)
        self._group_name_set.add(display_name.lower())
        for term in terms_sorted:
            self._group_term_lookup[term.lower()] = display_name
            self._set_item_group_state(term, display_name)
        self._refresh_group_list()
        self._update_group_button_enabled()
        self._update_action_button_text()

    def _edit_selected_group(self):
        item = self.groups_list.currentItem()
        if item is None:
            return
        name = item.data(Qt.UserRole) or ''
        group = next((grp for grp in self.groups if grp.get('name') == name), None)
        if group is None:
            return
        current_terms = list(dict.fromkeys(group.get('terms', []) or []))
        # Build term info list from existing items; include any missing terms from the group
        info_rows: List[dict] = []
        seen: Set[str] = set()
        for term, _item in self._item_by_term.items():
            if term in seen:
                continue
            seen.add(term)
            info_rows.append({
                'term': term,
                'frequency': self._term_frequency.get(term),
                'rank': None,
            })
        for term in current_terms:
            if term not in seen:
                seen.add(term)
                info_rows.append({'term': term, 'frequency': None, 'rank': None})

        dlg = TermSelectionDialog(
            self,
            info_rows,
            current_terms,
            window_title='Edit Group Members',
            info_text='Add or remove terms in this group. Terms already in other groups are disabled.',
            action_verb='Apply',
        )
        # Disable items belonging to other groups
        for i in range(dlg.list_widget.count()):
            it = dlg.list_widget.item(i)
            term_val = it.data(Qt.UserRole) or ''
            in_other = (term_val.lower() in self._group_term_lookup) and (self._group_term_lookup.get(term_val.lower()) != name)
            if in_other:
                it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
                txt = it.text()
                if '(in another group)' not in txt:
                    it.setText(f"{txt} (in another group)")
        if dlg.exec_() != QDialog.Accepted:
            return
        new_terms = list(dict.fromkeys(dlg.selected_terms))
        # Update mappings for removed terms
        removed = [t for t in current_terms if t not in new_terms]
        added = [t for t in new_terms if t not in current_terms]
        for t in removed:
            self._group_term_lookup.pop(t.lower(), None)
            self._set_item_group_state(t, None)
        for t in added:
            self._group_term_lookup[t.lower()] = name
            self._set_item_group_state(t, name)
        # Update group data
        freq_values = [self._term_frequency.get(term) for term in new_terms if self._term_frequency.get(term) is not None]
        total_freq = float(sum(freq_values)) if freq_values else None
        group['terms'] = new_terms
        if total_freq is not None:
            group['total_frequency'] = total_freq
        else:
            group.pop('total_frequency', None)
        # Update missing terms set
        missing_terms = [t for t in new_terms if t not in self._item_by_term]
        if missing_terms:
            group['missing_terms'] = list(missing_terms)
        else:
            group.pop('missing_terms', None)
        self._refresh_group_list()
        self._update_group_controls()
        self._update_action_button_text()

    def _rename_selected_group(self):
        item = self.groups_list.currentItem()
        if item is None:
            return
        name = item.data(Qt.UserRole) or ''
        group = next((grp for grp in self.groups if grp.get('name') == name), None)
        if group is None:
            return
        new_name, ok = QInputDialog.getText(self, 'Rename Group', 'Enter a new display name for this group:', QLineEdit.Normal, name)
        if not ok:
            return
        new_display = str(new_name).strip()
        if not new_display:
            QMessageBox.warning(self, 'Name Required', 'Enter a non-empty display name for the group.')
            return
        if new_display.lower() != name.lower() and new_display.lower() in self._group_name_set:
            QMessageBox.warning(self, 'Duplicate Name', 'A group with this name already exists. Choose a different name.')
            return
        self._group_name_set.discard(name.lower())
        self._group_name_set.add(new_display.lower())
        group['name'] = new_display
        for term in group.get('terms', []):
            if term.lower() in self._group_term_lookup:
                self._group_term_lookup[term.lower()] = new_display
            self._set_item_group_state(term, new_display)
        self._refresh_group_list()
        self._update_action_button_text()

    def _remove_selected_group(self):
        item = self.groups_list.currentItem()
        if item is None:
            return
        name = item.data(Qt.UserRole) or ''
        index = next((idx for idx, grp in enumerate(self.groups) if grp.get('name') == name), None)
        if index is None:
            return
        group = self.groups.pop(index)
        self._group_name_set.discard(name.lower())
        for term in group.get('terms', []):
            self._group_term_lookup.pop(term.lower(), None)
            self._set_item_group_state(term, None)
        self._refresh_group_list()
        self._update_group_button_enabled()
        self._update_action_button_text()

    def _set_item_group_state(self, term: str, group_name: Optional[str]):
        item = self._item_by_term.get(term)
        if item is None:
            return
        self.list_widget.blockSignals(True)
        if group_name:
            base_text = self._item_original_text.get(term, term)
            item.setText(f"{base_text} [Grouped → {group_name}]")
            item.setCheckState(Qt.Unchecked)
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
        else:
            base_text = self._item_original_text.get(term, term)
            item.setText(base_text)
            flags = item.flags()
            item.setFlags(flags | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
        self.list_widget.blockSignals(False)

    def _refresh_group_list(self):
        self.groups_list.blockSignals(True)
        self.groups_list.clear()
        for group in self.groups:
            label = self._format_group_description(group)
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, group.get('name'))
            self.groups_list.addItem(item)
        self.groups_list.blockSignals(False)
        self._update_group_controls()

    def _format_group_description(self, group: dict) -> str:
        name = group.get('name', '') or ''
        terms = group.get('terms', []) or []
        missing = {str(term).strip().lower() for term in group.get('missing_terms', []) or []}
        decorated_terms = []
        for term in terms:
            term_str = str(term)
            if term_str.lower() in missing:
                decorated_terms.append(f"{term_str} (not in list)")
            else:
                decorated_terms.append(term_str)
        terms_text = '; '.join(decorated_terms) if decorated_terms else '(none)'
        freq = group.get('total_frequency')
        freq_text = ''
        if isinstance(freq, (float, int)):
            value = float(freq)
            if abs(value - round(value)) < 1e-6:
                freq_text = f" (Total: {int(round(value)):,})"
            else:
                freq_text = f" (Total: {value:.2f})"
        return f"{name}: {terms_text}{freq_text}"

    def _update_group_controls(self):
        has_selection = self.groups_list.currentRow() >= 0
        if hasattr(self, 'edit_group_btn'):
            self.edit_group_btn.setEnabled(has_selection)
        self.rename_group_btn.setEnabled(has_selection)
        self.remove_group_btn.setEnabled(has_selection)

    def _update_group_button_enabled(self):
        self.group_btn.setEnabled(self._selected_terms_available())

    def _handle_item_changed(self, item):
        super()._handle_item_changed(item)
        self._update_group_button_enabled()

    def _clear_checks(self):
        super()._clear_checks()
        self._update_group_button_enabled()

    def _handle_length_selection(self, value: int):
        super()._handle_length_selection(value)
        self._clear_grouped_checks()
        self._update_group_button_enabled()

    def _clear_grouped_checks(self):
        self.list_widget.blockSignals(True)
        for term, item in self._item_by_term.items():
            if term.lower() in self._group_term_lookup:
                item.setCheckState(Qt.Unchecked)
        self.list_widget.blockSignals(False)

    def _update_action_button_text(self):
        group_count = len(self.groups)
        total_terms = sum(len(group.get('terms', []) or []) for group in self.groups)
        self.action_btn.setText(f'Create {group_count} Group(s) from {total_terms} Term(s)')
        self.action_btn.setEnabled(group_count > 0)

    def _accept_selection(self):
        def serialize(group: dict) -> dict:
            data = {
                'name': group.get('name'),
                'terms': list(group.get('terms', []) or []),
            }
            if group.get('total_frequency') is not None:
                data['total_frequency'] = float(group['total_frequency'])
            if group.get('missing_terms'):
                data['missing_terms'] = list(group.get('missing_terms'))
            return data

        self.created_groups = [serialize(group) for group in self.groups]
        self.accept()
class CollocationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Text Analysis')
        self.setMinimumSize(960, 720)
        self.resize(1200, 820)
        self.setSizeGripEnabled(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        self._last_output_paths = None
        self._last_topic_paths = None
        self._preview_windows = []
        self._collocate_map_settings: Optional[dict] = None
        self._rank_selected_terms: List[str] = []
        self._drop_section_button: Optional[QToolButton] = None
        self._selected_section_button: Optional[QToolButton] = None
        self._group_section_button: Optional[QToolButton] = None
        self._topic_section_button: Optional[QToolButton] = None
        self._drop_terms_prev_count = 0
        self._selected_terms_prev_count = 0
        self._group_terms_prev_count = 0
        self._rank_log_scale: bool = True
        self._operation_overlay: Optional[OperationOverlay] = None
        self._operation_worker: Optional[CancelableWorker] = None
        self._operation_context: Optional[str] = None
        self._operation_cancel_message: Optional[str] = None
        self._base_height: int = 0
        self._topic_trend_settings: Dict[str, Any] = {}

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(8)
        self.mode_json = QRadioButton('Use JSON results')
        self.mode_geo = QRadioButton('Use GeoJSON')
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.mode_geo)
        self.mode_group.addButton(self.mode_json)

        info_icon = QLabel()
        info_icon.setToolTip('JSON results run much faster than GeoJSON for collocation analysis.')
        info_icon.setPixmap(self.style().standardIcon(QStyle.SP_MessageBoxInformation).pixmap(16, 16))
        mode_row.addWidget(info_icon, 0, Qt.AlignVCenter)

        json_available = getattr(parent, 'json_file', None)
        geo_available = getattr(parent, 'geojson_file', None)
        if json_available:
            self.mode_json.setChecked(True)
        elif geo_available:
            self.mode_geo.setChecked(True)
        else:
            self.mode_json.setChecked(True)

        mode_row.addWidget(self.mode_json)
        mode_row.addWidget(self.mode_geo)

        self.choose_btn = QPushButton('Choose File…')
        mode_row.addWidget(self.choose_btn)

        top_controls = QWidget()
        top_controls_layout = QVBoxLayout(top_controls)
        top_controls_layout.setContentsMargins(0, 0, 0, 0)
        top_controls_layout.setSpacing(6)
        top_controls_layout.addLayout(mode_row)

        self.source_label = QLabel(self._source_text())
        self.source_label.setStyleSheet('font-weight: 600;')
        top_controls_layout.addWidget(self.source_label)
        layout.addWidget(top_controls)

        self.mode_geo.toggled.connect(lambda _: self.source_label.setText(self._source_text()))
        self.mode_json.toggled.connect(lambda _: self.source_label.setText(self._source_text()))
        self.mode_geo.toggled.connect(self.on_mode_toggle)
        self.mode_json.toggled.connect(self.on_mode_toggle)
        self.choose_btn.clicked.connect(self.choose_source_file)

        scroll_area = QScrollArea()
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(scroll_area, 1)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(18)

        form = QFormLayout()
        self.city_combo = QComboBox()
        self.city_combo.addItem('All Cities')
        city_row = QWidget()
        city_layout = QHBoxLayout(city_row)
        city_layout.setContentsMargins(0, 0, 0, 0)
        city_layout.addWidget(self.city_combo)
        form.addRow('City:', city_row)

        self.state_combo = QComboBox()
        self.state_combo.addItem('All States')
        state_row = QWidget()
        state_layout = QHBoxLayout(state_row)
        state_layout.setContentsMargins(0, 0, 0, 0)
        state_layout.addWidget(self.state_combo)
        form.addRow('State:', state_row)

        self.start_input = QLineEdit()
        self.end_input = QLineEdit()
        form.addRow('Start Date:', self.start_input)
        form.addRow('End Date:', self.end_input)

        self.term_input = QLineEdit()
        form.addRow('Search Term:', self.term_input)

        self.bin_size = QLineEdit('1')
        self.bin_size.setValidator(QIntValidator(1, 1000, self))
        self.bin_size.setMaximumWidth(60)
        form.addRow('Bin Size:', self.bin_size)
        self.bin_unit = QComboBox()
        self.bin_unit.addItems(['Days', 'Weeks', 'Months', 'Years'])
        self.bin_unit.setCurrentIndex(3)
        form.addRow('Time Unit:', self.bin_unit)
        self.ignore_bin = QCheckBox('Ignore Bin Size (no time binning)')
        self.ignore_bin.setChecked(True)
        form.addRow(self.ignore_bin)

        self._checkbox_order = [
            'include_page_count',
            'include_first_last_date',
            'include_cooccurrence_rate',
            'include_relative_position',
            'drop_stopwords',
            'write_occurrences_geojson',
        ]
        self._checkbox_labels = {
            'write_occurrences_geojson': 'Output occurrences GeoJSON',
        }
        self._checkbox_defaults = {opt: True for opt in self._checkbox_order}
        self._checkbox_defaults['write_occurrences_geojson'] = False
        self.checks = {}
        for opt in self._checkbox_order:
            cb = QCheckBox(self._checkbox_labels.get(opt, opt))
            cb.setChecked(self._checkbox_defaults.get(opt, True))
            self.checks[opt] = cb

        form_widget = QWidget()
        form_widget.setLayout(form)

        columns_layout = QHBoxLayout()
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(18)

        left_column = QVBoxLayout()
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(12)
        left_column.addWidget(form_widget)

        options_group = QGroupBox('Collocation Options')
        options_group.setAlignment(Qt.AlignLeft)
        options_group_layout = QVBoxLayout(options_group)
        options_group_layout.setContentsMargins(12, 12, 12, 12)
        for opt in self._checkbox_order:
            options_group_layout.addWidget(self.checks[opt])
        options_group_layout.addStretch(1)
        left_column.addWidget(options_group)

        context_group = QGroupBox('Context Window (words)')
        context_group.setAlignment(Qt.AlignLeft)
        context_layout = QHBoxLayout(context_group)
        context_layout.setContentsMargins(12, 12, 12, 12)
        context_layout.setSpacing(6)
        context_label = QLabel('Size:')
        context_layout.addWidget(context_label)
        self.context_left_spin = QSpinBox()
        self.context_left_spin.setRange(0, 99)
        self.context_left_spin.setValue(5)
        self.context_left_spin.setFixedWidth(46)
        context_layout.addWidget(self.context_left_spin)
        self.keyword_label = QLabel('<keyword>')
        self.keyword_label.setTextFormat(Qt.PlainText)
        self.keyword_label.setMinimumWidth(120)
        self.keyword_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        context_layout.addWidget(self.keyword_label, 1)
        self.context_right_spin = QSpinBox()
        self.context_right_spin.setRange(0, 99)
        self.context_right_spin.setValue(5)
        self.context_right_spin.setFixedWidth(46)
        context_layout.addWidget(self.context_right_spin)
        context_layout.addStretch(1)
        left_column.addWidget(context_group)

        left_column.addStretch(1)

        btn_run = QPushButton('Run Collocation')
        btn_bar = QPushButton('Show Bar Chart')
        btn_rank = QPushButton('Plot Collocate Rank Trends')
        btn_map_collocate = QPushButton('Create Collocate‑Rank Map')

        collocation_buttons_container = QWidget()
        collocation_buttons_layout = QVBoxLayout(collocation_buttons_container)
        collocation_buttons_layout.setContentsMargins(0, 0, 0, 0)
        collocation_buttons_layout.setSpacing(8)
        for button in (btn_run, btn_bar, btn_rank, btn_map_collocate):
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            collocation_buttons_layout.addWidget(button)

        drop_column = QVBoxLayout()
        drop_column.setContentsMargins(0, 0, 0, 0)
        drop_column.setSpacing(6)
        drop_buttons_row = QHBoxLayout()
        drop_buttons_row.setContentsMargins(0, 0, 0, 0)
        drop_buttons_row.setSpacing(6)
        self.select_drop_terms_btn = QPushButton('Drop Terms')
        self.select_drop_terms_btn.clicked.connect(self.open_drop_terms_dialog)
        self.select_drop_terms_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.add_custom_drop_btn = QPushButton('Add Custom…')
        self.add_custom_drop_btn.clicked.connect(self.add_custom_drop_terms)
        self.add_custom_drop_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        drop_buttons_row.addWidget(self.select_drop_terms_btn)
        drop_buttons_row.addWidget(self.add_custom_drop_btn)
        drop_column.addLayout(drop_buttons_row)

        drop_summary_row = QHBoxLayout()
        drop_summary_row.setContentsMargins(0, 0, 0, 0)
        self.drop_summary_label = QLabel()
        self.drop_summary_label.setWordWrap(True)
        self.drop_summary_label.setStyleSheet('color: #555555; font-size: 11px;')
        drop_summary_row.addWidget(self.drop_summary_label, 1)
        self.clear_drop_btn = QPushButton('Clear')
        self.clear_drop_btn.setFixedHeight(22)
        self.clear_drop_btn.setMaximumWidth(80)
        self.clear_drop_btn.clicked.connect(self.clear_dropped_terms)
        drop_summary_row.addWidget(self.clear_drop_btn, 0)
        drop_column.addLayout(drop_summary_row)

        self.drop_terms_view = QTextBrowser()
        self.drop_terms_view.setReadOnly(True)
        self.drop_terms_view.setMinimumHeight(140)
        self.drop_terms_view.setStyleSheet('font-size: 11px;')
        drop_column.addWidget(self.drop_terms_view)

        self.clear_notice_label = QLabel()
        self.clear_notice_label.setStyleSheet('color: #c05621; font-size: 11px;')
        self.clear_notice_label.hide()
        drop_column.addWidget(self.clear_notice_label)
        drop_column.addStretch(1)

        drop_content = QWidget()
        drop_content.setLayout(drop_column)
        drop_section, drop_toggle = self._create_collapsible_section('Drop Terms', drop_content, expanded=False)
        drop_section.setMaximumWidth(340)
        self._drop_section_button = drop_toggle

        group_column = QVBoxLayout()
        group_column.setContentsMargins(0, 0, 0, 0)
        group_column.setSpacing(6)
        self.group_terms_btn = QPushButton('Group Terms')
        self.group_terms_btn.clicked.connect(self.open_group_terms_dialog)
        self.group_terms_btn.setEnabled(False)
        self.group_terms_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        group_column.addWidget(self.group_terms_btn)

        group_summary_row = QHBoxLayout()
        group_summary_row.setContentsMargins(0, 0, 0, 0)
        self.group_summary_label = QLabel()
        self.group_summary_label.setWordWrap(True)
        self.group_summary_label.setStyleSheet('color: #555555; font-size: 11px;')
        group_summary_row.addWidget(self.group_summary_label, 1)
        self.clear_groups_btn = QPushButton('Clear')
        self.clear_groups_btn.setFixedHeight(22)
        self.clear_groups_btn.setMaximumWidth(80)
        self.clear_groups_btn.clicked.connect(self.clear_group_terms)
        group_summary_row.addWidget(self.clear_groups_btn, 0)
        group_column.addLayout(group_summary_row)

        self.group_terms_view = QTextBrowser()
        self.group_terms_view.setReadOnly(True)
        self.group_terms_view.setMinimumHeight(140)
        self.group_terms_view.setStyleSheet('font-size: 11px;')
        group_column.addWidget(self.group_terms_view)
        group_column.addStretch(1)

        group_content = QWidget()
        group_content.setLayout(group_column)
        group_section, group_toggle = self._create_collapsible_section('Group Terms', group_content, expanded=False)
        group_section.setMaximumWidth(340)
        self._group_section_button = group_toggle

        select_column = QVBoxLayout()
        select_column.setContentsMargins(0, 0, 0, 0)
        select_column.setSpacing(6)
        select_buttons_row = QHBoxLayout()
        select_buttons_row.setContentsMargins(0, 0, 0, 0)
        select_buttons_row.setSpacing(6)
        self.select_terms_btn = QPushButton('Select Terms')
        self.select_terms_btn.clicked.connect(self.open_select_terms_dialog)
        self.select_terms_btn.setEnabled(False)
        self.select_terms_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.add_custom_selected_btn = QPushButton('Add Custom…')
        self.add_custom_selected_btn.clicked.connect(self.add_custom_selected_terms)
        self.add_custom_selected_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        select_buttons_row.addWidget(self.select_terms_btn)
        select_buttons_row.addWidget(self.add_custom_selected_btn)
        select_column.addLayout(select_buttons_row)

        select_summary_row = QHBoxLayout()
        select_summary_row.setContentsMargins(0, 0, 0, 0)
        self.selected_summary_label = QLabel()
        self.selected_summary_label.setWordWrap(True)
        self.selected_summary_label.setStyleSheet('color: #555555; font-size: 11px;')
        select_summary_row.addWidget(self.selected_summary_label, 1)
        self.clear_selected_btn = QPushButton('Clear')
        self.clear_selected_btn.setFixedHeight(22)
        self.clear_selected_btn.setMaximumWidth(80)
        self.clear_selected_btn.clicked.connect(self.clear_selected_terms)
        select_summary_row.addWidget(self.clear_selected_btn, 0)
        select_column.addLayout(select_summary_row)

        self.selected_terms_view = QTextBrowser()
        self.selected_terms_view.setReadOnly(True)
        self.selected_terms_view.setMinimumHeight(140)
        self.selected_terms_view.setStyleSheet('font-size: 11px;')
        select_column.addWidget(self.selected_terms_view)
        select_column.addStretch(1)

        selected_content = QWidget()
        selected_content.setLayout(select_column)
        selected_section, selected_toggle = self._create_collapsible_section('Selected Terms', selected_content, expanded=False)
        selected_section.setMaximumWidth(340)
        self._selected_section_button = selected_toggle

        middle_column = QVBoxLayout()
        middle_column.setContentsMargins(0, 0, 0, 0)
        middle_column.setSpacing(12)
        middle_column.addWidget(drop_section)
        middle_column.addWidget(group_section)
        middle_column.addWidget(selected_section)
        middle_column.addStretch(1)

        topic_content = QWidget()
        topic_content_layout = QVBoxLayout(topic_content)
        topic_content_layout.setContentsMargins(12, 12, 12, 12)
        topic_content_layout.setSpacing(12)

        info_row_widget = QWidget(topic_content)
        info_row_layout = QHBoxLayout(info_row_widget)
        info_row_layout.setContentsMargins(0, 0, 0, 0)
        info_row_layout.setSpacing(6)
        topic_header = QLabel('Topic modeling settings')
        topic_header.setStyleSheet('font-weight: 600;')
        info_row_layout.addWidget(topic_header)
        topic_info_btn = QToolButton()
        topic_info_btn.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        topic_info_btn.setAutoRaise(True)
        topic_info_btn.setIconSize(QSize(16, 16))
        topic_info_btn.setStyleSheet('QToolButton { border: none; padding: 0; }')
        topic_info_btn.setToolTip('Topic modeling takes a long time on large corpora, and cancellation requests can take additional time to finish the current iteration.')
        info_row_layout.addWidget(topic_info_btn)
        info_row_layout.addStretch(1)
        topic_content_layout.addWidget(info_row_widget)

        topic_settings_form = QFormLayout()
        topic_settings_form.setLabelAlignment(Qt.AlignLeft)
        topic_settings_form.setFormAlignment(Qt.AlignLeft)
        topic_settings_form.setSpacing(6)

        self.topic_model_combo = QComboBox()
        self.topic_model_combo.addItems(['LDA', 'NMF'])
        self.topic_model_combo.setToolTip('Choose the topic modeling algorithm (LDA or NMF).')
        topic_settings_form.addRow('Model:', self.topic_model_combo)

        self.topic_topic_count_spin = QSpinBox()
        self.topic_topic_count_spin.setRange(2, 50)
        self.topic_topic_count_spin.setValue(10)
        self.topic_topic_count_spin.setToolTip('Number of latent topics to estimate.')
        topic_settings_form.addRow('Topics:', self.topic_topic_count_spin)

        self.topic_top_words_spin = QSpinBox()
        self.topic_top_words_spin.setRange(5, 50)
        self.topic_top_words_spin.setValue(12)
        self.topic_top_words_spin.setToolTip('How many of the most probable words to list for each topic.')
        topic_settings_form.addRow('Top words:', self.topic_top_words_spin)

        self.topic_max_features_spin = QSpinBox()
        self.topic_max_features_spin.setRange(500, 50000)
        self.topic_max_features_spin.setSingleStep(500)
        self.topic_max_features_spin.setValue(3000)
        self.topic_max_features_spin.setToolTip('Maximum vocabulary size for the vectorizer.')
        topic_settings_form.addRow('Max features:', self.topic_max_features_spin)

        self.topic_min_df_spin = QSpinBox()
        self.topic_min_df_spin.setRange(1, 100)
        self.topic_min_df_spin.setValue(5)
        self.topic_min_df_spin.setToolTip('Ignore tokens that appear in fewer documents than this threshold.')
        topic_settings_form.addRow('Min doc freq:', self.topic_min_df_spin)

        self.topic_max_df_spin = QDoubleSpinBox()
        self.topic_max_df_spin.setRange(0.05, 1.0)
        self.topic_max_df_spin.setSingleStep(0.05)
        self.topic_max_df_spin.setDecimals(2)
        self.topic_max_df_spin.setValue(0.5)
        self.topic_max_df_spin.setToolTip('Ignore tokens that appear in more than this fraction of documents.')
        topic_settings_form.addRow('Max doc freq:', self.topic_max_df_spin)

        self.topic_max_docs_spin = QSpinBox()
        self.topic_max_docs_spin.setRange(0, 100000)
        self.topic_max_docs_spin.setSingleStep(1000)
        self.topic_max_docs_spin.setSpecialValueText('All')
        self.topic_max_docs_spin.setValue(0)
        self.topic_max_docs_spin.setToolTip('Optional cap on the number of documents used for modeling.')
        topic_settings_form.addRow('Max documents:', self.topic_max_docs_spin)

        self.topic_min_weight_spin = QDoubleSpinBox()
        self.topic_min_weight_spin.setRange(0.0, 1.0)
        self.topic_min_weight_spin.setSingleStep(0.05)
        self.topic_min_weight_spin.setDecimals(2)
        self.topic_min_weight_spin.setValue(0.05)
        self.topic_min_weight_spin.setToolTip('Only keep topic assignments with at least this weight in each document.')
        topic_settings_form.addRow('Min topic weight:', self.topic_min_weight_spin)

        self.topic_doc_topics_spin = QSpinBox()
        self.topic_doc_topics_spin.setRange(1, 10)
        self.topic_doc_topics_spin.setValue(3)
        self.topic_doc_topics_spin.setToolTip('Limit how many topics are recorded for each document.')
        topic_settings_form.addRow('Topics per article:', self.topic_doc_topics_spin)

        topic_content_layout.addLayout(topic_settings_form)

        self.topic_restrict_selected_check = QCheckBox('Only include articles containing selected terms')
        self.topic_restrict_selected_check.setToolTip('Only analyze documents that contain the manually selected terms.')
        self.topic_require_selected_collocate_check = QCheckBox('Only include articles containing collocated selected term')
        self.topic_require_selected_collocate_check.setToolTip(
            'Keep only documents where the search term appears within the context window of at least one selected collocate.'
        )
        self.topic_exclude_drop_docs_check = QCheckBox('Exclude articles containing drop terms')
        self.topic_exclude_drop_docs_check.setToolTip('Skip documents that contain any drop terms.')
        self.topic_remove_drop_tokens_check = QCheckBox('Remove drop terms from tokenization')
        self.topic_remove_drop_tokens_check.setChecked(True)
        self.topic_remove_drop_tokens_check.setToolTip('Remove drop terms from the token list before modeling.')
        self.topic_include_url_check = QCheckBox('Include article URLs in topic documents CSV')
        self.topic_include_url_check.setToolTip('Add a column with article URLs to the topic_documents output CSV.')
        self.topic_include_url_check.setChecked(True)

        topic_options_group = QGroupBox('Topic Modeling Options')
        topic_options_group.setAlignment(Qt.AlignLeft)
        topic_options_layout = QVBoxLayout(topic_options_group)
        topic_options_layout.setContentsMargins(12, 12, 12, 12)
        for cb in (
            self.topic_restrict_selected_check,
            self.topic_require_selected_collocate_check,
            self.topic_exclude_drop_docs_check,
            self.topic_remove_drop_tokens_check,
            self.topic_include_url_check,
        ):
            topic_options_layout.addWidget(cb)
        topic_options_layout.addStretch(1)
        topic_content_layout.addWidget(topic_options_group)
        topic_content_layout.addStretch(1)

        topic_section, topic_toggle = self._create_collapsible_section(
            'Topic Modeling (beta)',
            topic_content,
            expanded=False,
            on_toggle=self._handle_topic_section_toggle,
        )
        self._topic_section_button = topic_toggle

        btn_topic = QPushButton('Run Topic Model')
        btn_topic_trends = QPushButton('Plot Topic Trends')
        self.btn_open_topic_docs = QPushButton('Open Topic Documents CSV')
        self.btn_open_topic_docs.setEnabled(False)

        topic_buttons_container = QWidget()
        topic_buttons_layout = QVBoxLayout(topic_buttons_container)
        topic_buttons_layout.setContentsMargins(0, 12, 0, 0)
        topic_buttons_layout.setSpacing(8)
        for button in (btn_topic, btn_topic_trends, self.btn_open_topic_docs):
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            topic_buttons_layout.addWidget(button)

        topic_column = QVBoxLayout()
        topic_column.setContentsMargins(0, 0, 0, 0)
        topic_column.setSpacing(12)
        topic_column.addWidget(topic_section)
        topic_column.addStretch(1)

        columns_layout.addLayout(left_column, 3)
        columns_layout.addLayout(middle_column, 3)
        columns_layout.addLayout(topic_column, 4)
        content_layout.addLayout(columns_layout)
        content_layout.addStretch(1)

        collocation_buttons_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        topic_buttons_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        buttons_panel = QWidget()
        buttons_layout = QHBoxLayout(buttons_panel)
        buttons_layout.setContentsMargins(0, 8, 0, 0)
        buttons_layout.setSpacing(24)
        buttons_layout.addWidget(collocation_buttons_container, 0, Qt.AlignLeft | Qt.AlignTop)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(topic_buttons_container, 0, Qt.AlignRight | Qt.AlignTop)
        layout.addWidget(buttons_panel, 0)

        btn_run.clicked.connect(self.run_collocate)
        btn_bar.clicked.connect(self.show_bar)
        btn_rank.clicked.connect(self.show_rank)
        btn_map_collocate.clicked.connect(self.create_collocate_rank_map)
        btn_topic.clicked.connect(self.run_topic_model_action)
        btn_topic_trends.clicked.connect(self.show_topic_trends)
        self.btn_open_topic_docs.clicked.connect(self.open_topic_documents_csv)

        self._update_selected_terms_summary()

        self._loading_defaults = True
        self._restore_state_or_defaults()
        self._loading_defaults = False
        self._update_drop_summary()
        self._update_group_summary()
        self._update_selected_terms_summary()
        self._update_context_keyword_label()
        self._set_clear_notice('')
        self._update_select_terms_enabled()
        self._update_topic_documents_button()

        self.bin_size.textEdited.connect(self._handle_bin_control_change)
        self.bin_unit.currentIndexChanged.connect(self._handle_bin_control_change)
        self.ignore_bin.stateChanged.connect(lambda *_: self._save_state())
        self.city_combo.currentIndexChanged.connect(lambda *_: self._save_state())
        self.state_combo.currentIndexChanged.connect(lambda *_: self._save_state())
        self.term_input.textEdited.connect(self._on_term_edited)
        self.start_input.textEdited.connect(lambda _text: self._save_state())
        self.end_input.textEdited.connect(lambda _text: self._save_state())
        self.mode_geo.toggled.connect(lambda _: self._save_state())
        self.mode_json.toggled.connect(lambda _: self._save_state())
        for cb in self.checks.values():
            cb.stateChanged.connect(lambda *_: self._save_state())
        self.context_left_spin.valueChanged.connect(lambda *_: self._save_state())
        self.context_right_spin.valueChanged.connect(lambda *_: self._save_state())
        self.topic_model_combo.currentIndexChanged.connect(lambda *_: self._save_state())
        self.topic_topic_count_spin.valueChanged.connect(lambda *_: self._save_state())
        self.topic_top_words_spin.valueChanged.connect(lambda *_: self._save_state())
        self.topic_max_features_spin.valueChanged.connect(lambda *_: self._save_state())
        self.topic_min_df_spin.valueChanged.connect(lambda *_: self._save_state())
        self.topic_max_df_spin.valueChanged.connect(lambda *_: self._save_state())
        self.topic_max_docs_spin.valueChanged.connect(lambda *_: self._save_state())
        self.topic_min_weight_spin.valueChanged.connect(lambda *_: self._save_state())
        self.topic_doc_topics_spin.valueChanged.connect(lambda *_: self._save_state())
        self.topic_restrict_selected_check.toggled.connect(self._handle_topic_restrict_selected_toggled)
        self.topic_require_selected_collocate_check.toggled.connect(self._handle_topic_require_collocate_toggled)
        self.topic_exclude_drop_docs_check.stateChanged.connect(lambda *_: self._save_state())
        self.topic_remove_drop_tokens_check.stateChanged.connect(lambda *_: self._save_state())
        self.topic_include_url_check.stateChanged.connect(lambda *_: self._save_state())

        if not self._base_height:
            self._base_height = self.sizeHint().height()
        self.setMinimumHeight(self._base_height)

    def _source_text(self):
        if self.mode_geo.isChecked():
            p = getattr(self.parent(), 'geojson_file', None)
            return f"GeoJSON: {os.path.basename(p) if p else '<none selected>'}"
        else:
            p = getattr(self.parent(), 'json_file', None)
        return f"JSON: {os.path.basename(p) if p else '<none selected>'}"

    def create_collocate_rank_map(self):
        parent = self.parent()
        if parent is None:
            QMessageBox.warning(self, 'Unavailable', 'Parent window not available.')
            return
        # Require a GeoJSON source to derive per‑city ranks
        geo_path = getattr(parent, 'geojson_file', None)
        if not geo_path or not os.path.exists(geo_path):
            # Try to prompt for a GeoJSON
            p, _ = QFileDialog.getOpenFileName(self, 'Select GeoJSON File', parent.project_folder or os.getcwd(), 'GeoJSON Files (*.geojson *.json)')
            if p:
                parent.geojson_file = p
                parent._update_loaded_file_labels()
                geo_path = p
        if not geo_path or not os.path.exists(geo_path):
            QMessageBox.warning(self, 'GeoJSON Required', 'Please add geographic info and select a GeoJSON file first.')
            return

        term = self.term_input.text().strip()
        if not term:
            QMessageBox.warning(self, 'Search Term Required', 'Enter a search term to build the collocate‑rank map.')
            return

        # Time configuration
        ignore_bin = self.ignore_bin.isChecked()
        size_text = self.bin_size.text().strip()
        bin_unit = self.bin_unit.currentText().lower()
        if not ignore_bin and (not size_text or not size_text.isdigit()):
            QMessageBox.warning(self, 'Invalid Bin Size', 'Please enter an integer ≥ 1.')
            return
        time_unit = bin_unit if not ignore_bin else 'month'
        time_step = int(size_text) if (not ignore_bin and size_text.isdigit()) else 1

        # Remember the requested date window for downstream map configuration
        start_value = self.start_input.text().strip()
        end_value = self.end_input.text().strip()

        # Collocation options influence tokenization
        opts = self._collect_options()
        drop_stop = bool(opts.get('drop_stopwords', False))
        context_left = self.context_left_spin.value()
        context_right = self.context_right_spin.value()
        context_window = max(context_left, context_right)

        # Gather geo metadata for settings dialog
        try:
            with open(geo_path, 'r', encoding='utf-8') as f:
                geo_payload = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, 'GeoJSON Error', f'Unable to read GeoJSON file:\n{exc}')
            return

        features = geo_payload.get('features') or []
        city_pairs: Set[Tuple[str, str]] = set()
        state_set: Set[str] = set()
        date_values: List[datetime] = []
        for feat in features:
            if not isinstance(feat, dict):
                continue
            props = feat.get('properties') or {}
            if not isinstance(props, dict):
                continue
            city_val = str(props.get('City') or '').strip()
            state_val = str(props.get('State') or '').strip()
            if city_val:
                city_pairs.add((city_val, state_val))
            if state_val:
                state_set.add(state_val)
            if not ignore_bin:
                dt = _parse_date(props.get('date'))
                if dt:
                    date_values.append(dt)

        time_bin_pairs: List[Tuple[str, str]] = []
        if not ignore_bin and date_values:
            try:
                index_list = _build_time_index(min(date_values), max(date_values), time_unit, max(1, time_step))
                for idx, dt in enumerate(index_list, start=1):
                    label = dt.strftime('%Y-%m-%d')
                    time_bin_pairs.append((str(idx), label))
            except Exception:
                time_bin_pairs = []

        sorted_cities = sorted(
            [pair for pair in city_pairs if pair[0]],
            key=lambda p: (p[0].lower(), p[1].lower()),
        )
        sorted_states = sorted(state_set, key=lambda s: s.lower())

        default_top = self._collocate_map_settings.get('top_n') if isinstance(self._collocate_map_settings, dict) else 10
        manual_terms_current = list(dict.fromkeys(self._rank_selected_terms))
        use_selected_default = bool(manual_terms_current)
        if isinstance(self._collocate_map_settings, dict) and 'use_selected_terms' in self._collocate_map_settings:
            use_selected_default = bool(self._collocate_map_settings.get('use_selected_terms')) and bool(manual_terms_current)
        settings_dialog = CollocateMapSettingsDialog(
            self,
            time_bins=time_bin_pairs,
            cities=sorted_cities,
            states=sorted_states,
            default_top_n=max(1, min(int(default_top or 10), 150)),
            max_top_n=150,
            selected_terms=manual_terms_current,
            use_selected_default=use_selected_default,
        )
        if self._collocate_map_settings:
            prev = self._collocate_map_settings
            prev_map_type = str(prev.get('map_type', 'rank')).lower()
            idx = settings_dialog.map_type_combo.findData(prev_map_type)
            if idx >= 0:
                settings_dialog.map_type_combo.setCurrentIndex(idx)
            settings_dialog._apply_map_type_constraints()
            settings_dialog.colorize_check.setChecked(bool(prev.get('colorize')))
            settings_dialog.lightweight_check.setChecked(bool(prev.get('lightweight', True)))
            settings_dialog.export_csv_check.setChecked(bool(prev.get('export_csv')))
            if prev.get('term_scope') == 'time' and settings_dialog.time_combo.count() > 0:
                desired = prev.get('time_key')
                if desired is not None:
                    for idx in range(settings_dialog.time_combo.count()):
                        if str(settings_dialog.time_combo.itemData(idx)) == str(desired):
                            settings_dialog.time_radio.setChecked(True)
                            settings_dialog.time_combo.setCurrentIndex(idx)
                            break
            if prev.get('location_scope') == 'city' and settings_dialog.city_combo.count() > 0:
                city_val = prev.get('location_city') or ''
                state_val = prev.get('location_state') or ''
                for idx in range(settings_dialog.city_combo.count()):
                    data = settings_dialog.city_combo.itemData(idx)
                    if isinstance(data, tuple) and data == (city_val, state_val):
                        settings_dialog.loc_city_radio.setChecked(True)
                        settings_dialog.city_combo.setCurrentIndex(idx)
                        break
            elif prev.get('location_scope') == 'state' and settings_dialog.state_combo.count() > 0:
                state_val = prev.get('location_state') or ''
                index = settings_dialog.state_combo.findText(state_val, Qt.MatchFixedString)
                if index >= 0:
                    settings_dialog.loc_state_radio.setChecked(True)
                    settings_dialog.state_combo.setCurrentIndex(index)
            if prev.get('enable_time_slider') and settings_dialog.enable_time_slider.isEnabled():
                settings_dialog.enable_time_slider.setChecked(True)
            if settings_dialog.use_selected_terms_check.isEnabled():
                settings_dialog.use_selected_terms_check.setChecked(bool(prev.get('use_selected_terms')))
            settings_dialog._apply_map_type_constraints()

        if settings_dialog.exec_() != QDialog.Accepted:
            return

        map_settings = settings_dialog.values()
        map_type = str(map_settings.get('map_type', 'rank') or 'rank').lower()
        self._collocate_map_settings = map_settings

        enable_time_slider = bool(map_settings.get('enable_time_slider')) and bool(time_bin_pairs)
        if not enable_time_slider:
            map_settings['enable_time_slider'] = False

        manual_terms = list(dict.fromkeys(self._rank_selected_terms))
        use_selected_terms = bool(map_settings.get('use_selected_terms')) if map_type != 'top_term' else False
        default_top_n = int(map_settings.get('top_n', 10))
        if use_selected_terms and manual_terms:
            metrics_path = None
            if isinstance(self._last_output_paths, dict):
                metrics_path = self._last_output_paths.get('metrics')
            if metrics_path and os.path.exists(metrics_path):
                try:
                    term_infos = self._fetch_metric_term_infos(metrics_path)
                except (RuntimeError, ValueError):
                    term_infos = []
                if term_infos:
                    available_terms = {info['term'] for info in term_infos}
                    missing_manual = [term for term in manual_terms if term not in available_terms]
                    if missing_manual and len(missing_manual) == len(manual_terms):
                        QMessageBox.information(
                            self,
                            'Terms Not Found',
                            'None of the selected terms are available in the current metrics. The map will use the Top N terms instead.',
                        )
                        manual_terms = []
                        use_selected_terms = False
                    elif missing_manual:
                        preview = ', '.join(missing_manual[:5])
                        if len(missing_manual) > 5:
                            preview += ', …'
                        QMessageBox.information(
                            self,
                            'Terms Not Found',
                            'The following selected terms are not available in the current metrics and will be skipped:\n' + preview,
                        )
                        manual_terms = [term for term in manual_terms if term in available_terms]
        else:
            use_selected_terms = False
        top_n = len(manual_terms) if use_selected_terms and manual_terms else default_top_n
        map_settings['use_selected_terms'] = use_selected_terms
        self._collocate_map_settings['use_selected_terms'] = use_selected_terms
        map_settings['map_type'] = map_type
        self._collocate_map_settings['map_type'] = map_type
        export_csv = bool(map_settings.get('export_csv'))
        map_settings['export_csv'] = export_csv
        self._collocate_map_settings['export_csv'] = export_csv
        term_scope = map_settings.get('term_scope', 'global')
        time_key = map_settings.get('time_key') or None
        time_label = map_settings.get('time_label') or ''
        location_scope = map_settings.get('location_scope', 'all')
        location_city = map_settings.get('location_city') or ''
        location_state = map_settings.get('location_state') or ''
        location_label = map_settings.get('location_label') or ''

        lightweight_mode = bool(map_settings.get('lightweight', True))
        map_settings['lightweight'] = lightweight_mode
        self._collocate_map_settings['lightweight'] = lightweight_mode

        def task(*, cancel_event: Optional[threading.Event]):
            return create_map(
                geo_path,
                mode='points',
                time_unit=time_unit,
                time_step=time_step,
                linger_unit='week',
                linger_step=0,
                disable_time=not enable_time_slider,
                lightweight=lightweight_mode,
                table_mode='minimal',
                table_row_limit=1000,
                collocate_rank_mode=True,
                collocate_drop_stopwords=drop_stop,
                collocate_drop_terms=self._get_parent_drop_terms(),
                collocate_term_groups=self._get_parent_term_groups(),
                collocate_rank_top_n=top_n,
                collocate_rank_term_scope=term_scope,
                collocate_rank_time_key=time_key,
                collocate_rank_focus=location_scope,
                collocate_rank_focus_city=location_city or None,
                collocate_rank_focus_state=location_state or None,
                collocate_rank_terms=manual_terms if use_selected_terms and manual_terms else None,
                collocate_window=context_window,
                collocate_rank_time_label=time_label or None,
                collocate_rank_focus_label=location_label or None,
                collocate_rank_colorize=bool(map_settings.get('colorize')),
                collocate_time_slider=enable_time_slider,
                collocate_map_variant=map_type,
                collocate_search_term=term,
                metadata_enabled=getattr(parent, 'metadata_enabled', True),
                project_dir=parent.project_folder if parent else None,
                time_start_override=start_value or None,
                time_end_override=end_value or None,
                collocate_export_csv=export_csv,
                cancel_event=cancel_event,
            )

        def handle_success(result: Dict[str, Optional[str]]):
            map_path = (result or {}).get('map_path')
            if map_path and os.path.exists(map_path):
                self._set_clear_notice('')
                encoded = urllib.parse.quote(map_path)
                log_lines = [
                    f'<div>Created map: <a href="chronam-open:{encoded}">{html.escape(map_path)}</a></div>'
                ]
                map_type_display = 'Top Ranked Term by Location' if map_type == 'top_term' else 'Select Term Rank Map'
                log_lines.append(f'<div>Map type: {html.escape(map_type_display)}</div>')
                csv_export_path = (result or {}).get('collocate_csv')
                if csv_export_path and os.path.exists(csv_export_path):
                    encoded_csv = urllib.parse.quote(csv_export_path)
                    log_lines.append(
                        f'<div>Collocate CSV: <a href="chronam-open:{encoded_csv}">{html.escape(csv_export_path)}</a></div>'
                    )
                if parent and hasattr(parent, 'append_project_log'):
                    parent.append_project_log('Collocate‑Rank Map', log_lines)
                QDesktopServices.openUrl(QUrl.fromLocalFile(map_path))
            else:
                QMessageBox.information(self, 'No Map Created', 'The map was not created.')

        def handle_error(exc: Exception):
            QMessageBox.critical(self, 'Map Error', str(exc))

        def handle_cancel():
            self._set_clear_notice('Map build cancelled.')

        started = self._start_operation(
            'Building collocate map…',
            task,
            on_success=handle_success,
            on_error=handle_error,
            on_cancel=handle_cancel,
            context='collocate-map',
        )
        if started:
            return
        return

    def choose_source_file(self):
        parent = self.parent()
        if self.mode_geo.isChecked():
            start_dir = parent.project_folder if parent else os.getcwd()
            processed_dir = os.path.join(start_dir, 'data', 'processed')
            if os.path.isdir(processed_dir):
                start_dir = processed_dir
            p, _ = QFileDialog.getOpenFileName(self, 'Select GeoJSON File', start_dir, 'GeoJSON Files (*.geojson *.json)')
            if p:
                parent.geojson_file = p
                parent._update_loaded_file_labels()
                # Update city/state lists for new GeoJSON
                self.populate_city_state()
        else:
            start_dir = parent.project_folder if parent else os.getcwd()
            processed_dir = os.path.join(start_dir, 'data', 'processed')
            if os.path.isdir(processed_dir):
                start_dir = processed_dir
            p, _ = QFileDialog.getOpenFileName(self, 'Select JSON Results', start_dir, 'JSON Files (*.json)')
            if p:
                parent.json_file = p
                parent._update_loaded_file_labels()
        # Update source label text
        self.source_label.setText(self._source_text())
        if parent is not None:
                parent.collocation_state = {'dropped_terms': [], 'term_groups': [], 'topic_settings': {}, 'topic_trend_settings': {}}
        self._last_output_paths = None
        self._rank_selected_terms = []
        self._update_selected_terms_summary()
        self._update_select_terms_enabled()
        self._loading_defaults = True
        self._prefill_from_current_source(reset_state=True)
        self._loading_defaults = False
        self._save_state()

    def _restore_state_or_defaults(self):
        state = {}
        parent = self.parent()
        if parent is not None:
            state = getattr(parent, 'collocation_state', {}) or {}
        if state:
            self._apply_state(state)
        else:
            # Prefer JSON by default if available; else GeoJSON; else JSON
            json_available = getattr(parent, 'json_file', None) if parent is not None else None
            geo_available = getattr(parent, 'geojson_file', None) if parent is not None else None
            if json_available and not geo_available:
                self.mode_json.setChecked(True)
            elif geo_available and not json_available:
                self.mode_geo.setChecked(True)
            elif json_available and geo_available:
                self.mode_json.setChecked(True)
            else:
                self.mode_json.setChecked(True)
            self.on_mode_toggle()
            self._prefill_from_current_source()
        self.source_label.setText(self._source_text())

    def _apply_state(self, state: dict):
        parent = self.parent()
        mode = state.get('mode', 'geo')
        self.mode_geo.blockSignals(True)
        self.mode_json.blockSignals(True)
        json_available = getattr(parent, 'json_file', None) if parent is not None else None
        geo_available = getattr(parent, 'geojson_file', None) if parent is not None else None
        chosen = None
        if mode == 'json' and json_available:
            chosen = 'json'
        elif mode == 'geo' and geo_available:
            chosen = 'geo'
        elif json_available and not geo_available:
            chosen = 'json'
        elif geo_available and not json_available:
            chosen = 'geo'
        elif json_available:
            chosen = 'json'
        elif geo_available:
            chosen = 'geo'
        if chosen == 'geo':
            self.mode_geo.setChecked(True)
        else:
            self.mode_json.setChecked(True)
        self.mode_geo.blockSignals(False)
        self.mode_json.blockSignals(False)
        self.on_mode_toggle()

        # Prefill term/date from current source if missing in saved state
        self._prefill_from_current_source()
        self.source_label.setText(self._source_text())

        city = state.get('city')
        if city:
            idx = self.city_combo.findText(city, Qt.MatchFixedString)
            if idx == -1 and city not in ('All Cities', ''):
                self.city_combo.addItem(city)
                idx = self.city_combo.count() - 1
            if idx >= 0:
                self.city_combo.setCurrentIndex(idx)
        state_val = state.get('state')
        if state_val:
            idx = self.state_combo.findText(state_val, Qt.MatchFixedString)
            if idx == -1 and state_val not in ('All States', ''):
                self.state_combo.addItem(state_val)
                idx = self.state_combo.count() - 1
            if idx >= 0:
                self.state_combo.setCurrentIndex(idx)

        self.term_input.setText(state.get('term', self.term_input.text()))
        self._update_context_keyword_label()
        self.start_input.setText(state.get('start', self.start_input.text()))
        self.end_input.setText(state.get('end', self.end_input.text()))

        bin_size = state.get('bin_size')
        if bin_size:
            self.bin_size.setText(str(bin_size))
        bin_unit = state.get('bin_unit')
        if bin_unit:
            idx = self.bin_unit.findText(bin_unit, Qt.MatchFixedString)
            if idx >= 0:
                self.bin_unit.setCurrentIndex(idx)
        ignore = state.get('ignore_bin')
        if ignore is not None:
            self.ignore_bin.setChecked(bool(ignore))

        opts = state.get('options', {})
        for key, cb in self.checks.items():
            default = self._checkbox_defaults.get(key, cb.isChecked())
            cb.setChecked(bool(opts.get(key, default)))

        context_left = state.get('context_left')
        if context_left is not None:
            try:
                self.context_left_spin.setValue(int(context_left))
            except (TypeError, ValueError):
                pass
        context_right = state.get('context_right')
        if context_right is not None:
            try:
                self.context_right_spin.setValue(int(context_right))
            except (TypeError, ValueError):
                pass

        drop_terms_state = state.get('dropped_terms')
        if parent is not None:
            if isinstance(drop_terms_state, list):
                parent.collocation_drop_terms = [str(term).strip() for term in drop_terms_state if isinstance(term, str) and term.strip()]
            elif drop_terms_state is None:
                # Leave as-is when state does not specify dropped terms
                pass
            else:
                parent.collocation_drop_terms = []
        self._update_drop_summary()

        groups_state = state.get('term_groups')
        if groups_state is not None:
            self._set_term_groups(groups_state, log_change=False)
        else:
            self._update_group_summary()

        manual_terms_state = state.get('rank_selected_terms')
        if isinstance(manual_terms_state, list):
            cleaned: List[str] = []
            seen = set()
            for entry in manual_terms_state:
                if not isinstance(entry, str):
                    continue
                term_clean = entry.strip()
                if term_clean and term_clean not in seen:
                    seen.add(term_clean)
                    cleaned.append(term_clean)
            self._rank_selected_terms = cleaned
        else:
            self._rank_selected_terms = []
        self._rank_log_scale = bool(state.get('rank_log_scale', True))
        topic_settings = state.get('topic_settings') or state.get('topic')
        if isinstance(topic_settings, dict):
            self._apply_topic_state(topic_settings)
        else:
            self._update_topic_toggle_states()
        trend_settings = state.get('topic_trend_settings')
        if isinstance(trend_settings, dict):
            self._topic_trend_settings = dict(trend_settings)
        else:
            self._topic_trend_settings = {}

    def _prefill_from_current_source(self, reset_state: bool = False):
        parent = self.parent()
        source_path = None
        use_geo = self.mode_geo.isChecked()
        if use_geo:
            source_path = getattr(parent, 'geojson_file', None)
        else:
            source_path = getattr(parent, 'json_file', None)

        meta = self._extract_metadata_from_source(source_path, use_geo) if source_path else {}
        if reset_state:
            # When switching sources, reset combos before populating
            if use_geo:
                self.populate_city_state()
            else:
                self.city_combo.setCurrentIndex(0)
                self.state_combo.setCurrentIndex(0)

        if meta.get('term'):
            self.term_input.setText(meta.get('term'))
            self._update_context_keyword_label()
        if meta.get('start_date'):
            self.start_input.setText(meta.get('start_date'))
        if meta.get('end_date'):
            self.end_input.setText(meta.get('end_date'))
        self._update_context_keyword_label()

    def _extract_metadata_from_source(self, path: Optional[str], is_geo: bool) -> dict:
        result = {'term': '', 'start_date': '', 'end_date': ''}
        if not path or not os.path.exists(path):
            return result
        try:
            if is_geo:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                meta = data.get('metadata') or {}
                if not meta and data.get('properties'):
                    meta = data.get('properties', {})
                if not meta:
                    features = data.get('features', [])
                    if features:
                        probe = features[0].get('properties', {}) or {}
                        meta = {
                            'search_term': probe.get('search_term') or probe.get('SearchTerm'),
                            'start_date': probe.get('start_date') or probe.get('StartDate'),
                            'end_date': probe.get('end_date') or probe.get('EndDate'),
                        }
                result['term'] = meta.get('search_term') or meta.get('term') or ''
                result['start_date'] = meta.get('start_date') or ''
                result['end_date'] = meta.get('end_date') or ''
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                result['term'] = info.get('search_term', '')
                result['start_date'] = info.get('start_date', '')
                result['end_date'] = info.get('end_date', '')
        except Exception:
            return self._parse_filename_metadata(path)

        if not (result['term'] and result['start_date'] and result['end_date']):
            fallback = self._parse_filename_metadata(path)
            for key in result:
                if not result[key] and fallback.get(key):
                    result[key] = fallback[key]
        return result

    def _parse_filename_metadata(self, path: str) -> dict:
        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', name)
        start = dates[0] if len(dates) >= 1 else ''
        end = dates[1] if len(dates) >= 2 else ''
        term_part = name
        if start:
            term_part = term_part.split(start)[0].rstrip('_-')
        term = term_part.split('_')[-1] if term_part else ''
        for prefix in ['merged', 'collocates', 'occurrences']:
            if term == prefix and '_' in term_part:
                term = term_part.split('_')[-2]
        return {'term': term, 'start_date': start, 'end_date': end}

    def _handle_bin_control_change(self, *_):
        if self._loading_defaults:
            return
        if self.ignore_bin.isChecked():
            self.ignore_bin.blockSignals(True)
            self.ignore_bin.setChecked(False)
            self.ignore_bin.blockSignals(False)
        self._save_state()

    def _collect_options(self) -> dict:
        return {opt: cb.isChecked() for opt, cb in self.checks.items()}

    def _options_with_context(self) -> dict:
        opts = self._collect_options()
        opts = dict(opts)
        opts['window_left'] = self.context_left_spin.value()
        opts['window_right'] = self.context_right_spin.value()
        return opts

    def _collect_topic_settings(self) -> dict:
        return {
            'model': self.topic_model_combo.currentText().strip().lower(),
            'n_topics': self.topic_topic_count_spin.value(),
            'n_top_words': self.topic_top_words_spin.value(),
            'max_features': self.topic_max_features_spin.value(),
            'min_df': self.topic_min_df_spin.value(),
            'max_df': round(self.topic_max_df_spin.value(), 2),
            'max_documents': self.topic_max_docs_spin.value(),
            'min_topic_weight': round(self.topic_min_weight_spin.value(), 2),
            'max_topics_per_document': self.topic_doc_topics_spin.value(),
            'restrict_selected': self.topic_restrict_selected_check.isChecked(),
            'require_selected_collocate': self.topic_require_selected_collocate_check.isChecked(),
            'exclude_drop_docs': self.topic_exclude_drop_docs_check.isChecked(),
            'remove_drop_tokens': self.topic_remove_drop_tokens_check.isChecked(),
            'include_article_url': self.topic_include_url_check.isChecked(),
            'collocate_window_left': self.context_left_spin.value(),
            'collocate_window_right': self.context_right_spin.value(),
        }


    def _topic_parameters(self, drop_terms: Sequence[str], selected_terms: Sequence[str], *, settings: Optional[dict] = None) -> TopicModelParameters:
        if settings is None:
            settings = self._collect_topic_settings()
        max_docs_val = int(settings.get('max_documents') or 0)
        drop_stopwords = bool(self.checks.get('drop_stopwords') and self.checks['drop_stopwords'].isChecked())
        raw_min_df = settings.get('min_df', 5)
        try:
            min_df_numeric = float(raw_min_df)
        except (TypeError, ValueError):
            min_df_numeric = 5.0
        if min_df_numeric >= 1.0 and abs(min_df_numeric - round(min_df_numeric)) < 1e-9:
            min_df_value = int(round(min_df_numeric))
        else:
            min_df_value = max(0.0, min_df_numeric)

        raw_max_df = settings.get('max_df', 0.5)
        try:
            max_df_numeric = float(raw_max_df)
        except (TypeError, ValueError):
            max_df_numeric = 0.5
        if max_df_numeric <= 0.0:
            max_df_numeric = 0.05
        if max_df_numeric > 1.0:
            max_df_numeric = 1.0
        left_default = self.context_left_spin.value() if hasattr(self, 'context_left_spin') else 5
        right_default = self.context_right_spin.value() if hasattr(self, 'context_right_spin') else 5

        def _safe_window(value: Any, default: int) -> int:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return max(0, int(default))

        window_left_val = _safe_window(settings.get('collocate_window_left'), left_default)
        window_right_val = _safe_window(settings.get('collocate_window_right'), right_default)

        params = TopicModelParameters(
            model=str(settings.get('model', 'lda') or 'lda'),
            n_topics=int(settings.get('n_topics', 10)),
            n_top_words=int(settings.get('n_top_words', 12)),
            max_features=int(settings.get('max_features', 3000)),
            min_df=min_df_value,
            max_df=max_df_numeric,
            max_documents=max_docs_val if max_docs_val > 0 else None,
            drop_stopwords=drop_stopwords,
            restrict_to_selected_terms=bool(settings.get('restrict_selected') and selected_terms),
            require_selected_collocate=bool(
                settings.get('require_selected_collocate')
                and settings.get('restrict_selected')
                and selected_terms
            ),
            exclude_drop_term_documents=bool(settings.get('exclude_drop_docs') and drop_terms),
            remove_drop_terms_from_tokens=bool(settings.get('remove_drop_tokens') and drop_terms),
            min_topic_weight=float(settings.get('min_topic_weight', 0.05)),
            max_topics_per_document=int(settings.get('max_topics_per_document', 3)),
            collocate_window_left=window_left_val,
            collocate_window_right=window_right_val,
        )
        return params

    def _build_topic_output_paths_topic(
        self,
        term: str,
        start: Optional[str],
        end: Optional[str],
        city: Optional[str],
        state: Optional[str],
        params: TopicModelParameters,
        selected_terms: Sequence[str],
    ):
        parent = self.parent()
        if parent is None:
            raise RuntimeError('Collocation dialog has no parent window')
        time_bin_unit = self._current_time_bin_unit()
        return build_topic_model_output_paths(
            parent.project_folder,
            term=term,
            start_date=start or None,
            end_date=end or None,
            city=city,
            state=state,
            time_bin_unit=time_bin_unit,
            ignore_bin=self.ignore_bin.isChecked(),
            params=params,
            drop_terms=self._get_parent_drop_terms(),
            term_groups=self._get_parent_term_groups(),
            selected_terms=selected_terms,
        )

    def _apply_topic_state(self, settings: dict):
        model = str(settings.get('model', 'lda') or 'lda').lower()
        model_label = 'LDA' if model == 'lda' else 'NMF'
        idx = self.topic_model_combo.findText(model_label, Qt.MatchFixedString)
        if idx < 0:
            idx = 0
        self.topic_model_combo.setCurrentIndex(idx)

        def _set_int_spin(spin: QSpinBox, value: Any):
            try:
                if value is None:
                    return
                spin.setValue(int(value))
            except (TypeError, ValueError):
                pass

        def _set_double_spin(spin: QDoubleSpinBox, value: Any):
            try:
                if value is None:
                    return
                spin.setValue(float(value))
            except (TypeError, ValueError):
                pass

        _set_int_spin(self.topic_topic_count_spin, settings.get('n_topics'))
        _set_int_spin(self.topic_top_words_spin, settings.get('n_top_words'))
        _set_int_spin(self.topic_max_features_spin, settings.get('max_features'))
        _set_int_spin(self.topic_min_df_spin, settings.get('min_df'))
        _set_double_spin(self.topic_max_df_spin, settings.get('max_df'))
        max_docs = settings.get('max_documents')
        if max_docs in (None, '', 0):
            self.topic_max_docs_spin.setValue(0)
        else:
            _set_int_spin(self.topic_max_docs_spin, max_docs)
        _set_double_spin(self.topic_min_weight_spin, settings.get('min_topic_weight'))
        _set_int_spin(self.topic_doc_topics_spin, settings.get('max_topics_per_document'))

        self.topic_restrict_selected_check.setChecked(bool(settings.get('restrict_selected')))
        self.topic_require_selected_collocate_check.setChecked(bool(settings.get('require_selected_collocate')))
        self.topic_exclude_drop_docs_check.setChecked(bool(settings.get('exclude_drop_docs')))
        self.topic_remove_drop_tokens_check.setChecked(bool(settings.get('remove_drop_tokens', True)))
        self.topic_include_url_check.setChecked(bool(settings.get('include_article_url', True)))
        self._update_topic_toggle_states()

    def _update_topic_toggle_states(self):
        has_selected = bool(self._rank_selected_terms)
        if not has_selected and self.topic_restrict_selected_check.isChecked():
            self.topic_restrict_selected_check.blockSignals(True)
            self.topic_restrict_selected_check.setChecked(False)
            self.topic_restrict_selected_check.blockSignals(False)
        self.topic_restrict_selected_check.setEnabled(has_selected)

        allow_collocate_filter = has_selected and self.topic_restrict_selected_check.isChecked()
        if not allow_collocate_filter and self.topic_require_selected_collocate_check.isChecked():
            self.topic_require_selected_collocate_check.blockSignals(True)
            self.topic_require_selected_collocate_check.setChecked(False)
            self.topic_require_selected_collocate_check.blockSignals(False)
        self.topic_require_selected_collocate_check.setEnabled(allow_collocate_filter)

        has_drop = bool(self._get_parent_drop_terms())
        if not has_drop and self.topic_exclude_drop_docs_check.isChecked():
            self.topic_exclude_drop_docs_check.blockSignals(True)
            self.topic_exclude_drop_docs_check.setChecked(False)
            self.topic_exclude_drop_docs_check.blockSignals(False)
        self.topic_exclude_drop_docs_check.setEnabled(has_drop)

        if not has_drop:
            self.topic_remove_drop_tokens_check.blockSignals(True)
            self.topic_remove_drop_tokens_check.setChecked(False)
            self.topic_remove_drop_tokens_check.blockSignals(False)
            self.topic_remove_drop_tokens_check.setEnabled(False)
        else:
            self.topic_remove_drop_tokens_check.setEnabled(True)

    def _handle_topic_restrict_selected_toggled(self, checked: bool):
        if not checked and self.topic_require_selected_collocate_check.isChecked():
            self.topic_require_selected_collocate_check.blockSignals(True)
            self.topic_require_selected_collocate_check.setChecked(False)
            self.topic_require_selected_collocate_check.blockSignals(False)
        self._update_topic_toggle_states()
        self._save_state()

    def _handle_topic_require_collocate_toggled(self, checked: bool):
        if checked and not self.topic_restrict_selected_check.isChecked():
            self.topic_restrict_selected_check.blockSignals(True)
            self.topic_restrict_selected_check.setChecked(True)
            self.topic_restrict_selected_check.blockSignals(False)
        self._update_topic_toggle_states()
        self._save_state()

    def _handle_topic_section_toggle(self, checked: bool):
        if not self._base_height:
            self._base_height = self.minimumHeight() or self.sizeHint().height()
        if not checked:
            self.setMinimumHeight(self._base_height)

    def _start_operation(
        self,
        message: str,
        task: Callable[..., Any],
        *,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        context: Optional[str] = None,
        cancel_requested_message: Optional[str] = None,
    ) -> bool:
        if self._operation_overlay is not None:
            QMessageBox.information(self, 'Busy', 'Another task is already running. Please wait for it to finish.')
            return False
        overlay = OperationOverlay(self, message)
        worker = CancelableWorker(task)
        overlay.cancel_requested.connect(lambda: self._handle_operation_cancel(worker, overlay))
        worker.finished.connect(lambda result, cancelled: self._handle_operation_finished(result, cancelled, on_success, on_cancel))
        worker.error.connect(lambda exc: self._handle_operation_error(exc, on_error))
        self._operation_overlay = overlay
        self._operation_worker = worker
        self._operation_context = context
        self._operation_cancel_message = cancel_requested_message
        overlay.show_overlay()
        worker.start()
        return True

    def _handle_operation_cancel(self, worker: CancelableWorker, overlay: OperationOverlay):
        if worker is not None:
            worker.request_cancel()
        if overlay is not None:
            overlay.mark_cancelled()
        if self._operation_cancel_message:
            self._set_clear_notice(self._operation_cancel_message)

    def _handle_operation_finished(
        self,
        result: Any,
        cancelled: bool,
        on_success: Optional[Callable[[Any], None]],
        on_cancel: Optional[Callable[[], None]],
    ):
        self._cleanup_operation()
        if cancelled:
            if on_cancel:
                on_cancel()
            else:
                self._set_clear_notice('Operation cancelled.')
            return
        if on_success:
            on_success(result)

    def _handle_operation_error(self, exc: Exception, on_error: Optional[Callable[[Exception], None]]):
        self._cleanup_operation()
        if on_error:
            on_error(exc)
        else:
            QMessageBox.critical(self, 'Error', str(exc))

    def _cleanup_operation(self):
        overlay = self._operation_overlay
        worker = self._operation_worker
        self._operation_overlay = None
        self._operation_worker = None
        self._operation_context = None
        self._operation_cancel_message = None
        if worker is not None:
            worker.wait(50)
            worker.deleteLater()
        if overlay is not None:
            overlay.close_overlay()
    def _generate_by_time_csv(
        self,
        *,
        city: Optional[str],
        state: Optional[str],
        start_value: Optional[str],
        end_value: Optional[str],
        term: str,
        time_bin_unit: Optional[str],
        drop_terms: List[str],
        options_runtime: dict,
        options_hash: dict,
        window_left: int,
        window_right: int,
        metadata_enabled: bool,
        prefer_geo: bool,
        cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        parent = self.parent()
        if parent is None:
            return None, 'Parent window not available.'
        json_path = getattr(parent, 'json_file', None)
        geo_path = getattr(parent, 'geojson_file', None)
        candidates = []
        if prefer_geo:
            candidates.extend([(geo_path, True), (json_path, False)])
        else:
            candidates.extend([(json_path, False), (geo_path, True)])
        options_runtime = dict(options_runtime)
        source_path = None
        source_is_geo = False
        for candidate, is_geo in candidates:
            if candidate and os.path.exists(candidate):
                source_path = candidate
                source_is_geo = is_geo
                break
        if not source_path:
            return None, 'Locate a JSON or GeoJSON results file before building rank changes.'
        if not source_is_geo:
            options_runtime['write_occurrences_geojson'] = False
        result = run_collocation(
            parent.project_folder,
            city=city,
            state=state,
            start_date=start_value or None,
            end_date=end_value or None,
            term=term,
            time_bin_unit=time_bin_unit,
            json_path=None if source_is_geo else source_path,
            geojson_path=source_path if source_is_geo else None,
            ignore_bin=self.ignore_bin.isChecked(),
            write_by_time=True,
            drop_terms=drop_terms,
            term_groups=self._get_parent_term_groups(),
            window_left=window_left,
            window_right=window_right,
            metadata_enabled=metadata_enabled,
            write_metrics=False,
            cancel_event=cancel_event,
            **options_runtime,
        )
        built_path = result.get('by_time') if isinstance(result, dict) else None
        if built_path and os.path.exists(built_path):
            return built_path, None
        predicted = self._build_output_paths(term, start_value, end_value, city, state, options_hash)
        expected = predicted.get('by_time')
        if expected and os.path.exists(expected):
            return expected, None
        return None, 'By-time data could not be created. Please run the collocation analysis first.'

    def _derive_time_bins_from_inputs(self) -> List[str]:
        start_text = self.start_input.text().strip()
        end_text = self.end_input.text().strip()
        bin_unit = self._current_time_bin_unit()

        if not start_text or not end_text or not bin_unit:
            return []

        try:
            start_dt = pd.to_datetime(start_text, errors='raise')
            end_dt = pd.to_datetime(end_text, errors='raise')
        except Exception:
            return []

        if pd.isna(start_dt) or pd.isna(end_dt):
            return []

        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt

        parts = bin_unit.strip().split()
        if len(parts) != 2:
            return []
        try:
            size = int(parts[0])
        except ValueError:
            return []
        unit = parts[1].lower()
        if size <= 0:
            return []

        current = start_dt
        end_inclusive = end_dt
        bins: List[str] = []
        max_bins = 600

        while current <= end_inclusive and len(bins) < max_bins:
            bins.append(current.date().isoformat())
            if unit.startswith('day'):
                current = current + pd.Timedelta(days=size)
            elif unit.startswith('week'):
                current = current + pd.Timedelta(weeks=size)
            elif unit.startswith('month'):
                current = current + pd.DateOffset(months=size)
            elif unit.startswith('year'):
                current = current + pd.DateOffset(years=size)
            else:
                return []

        if not bins:
            bins.append(start_dt.date().isoformat())
        return bins

    def _update_context_keyword_label(self):
        if not hasattr(self, 'keyword_label'):
            return
        term_text = self.term_input.text().strip()
        display = term_text if term_text else '<keyword>'
        self.keyword_label.setText(display)

    def _on_term_edited(self, _text: str):
        self._update_context_keyword_label()
        self._save_state()

    def _get_parent_drop_terms(self) -> List[str]:
        parent = self.parent()
        if parent is None:
            return []
        return list(getattr(parent, 'collocation_drop_terms', []))

    def _get_parent_term_groups(self) -> List[dict]:
        parent = self.parent()
        if parent is None:
            return []
        groups = getattr(parent, 'collocation_term_groups', [])
        cleaned: List[dict] = []
        for entry in groups or []:
            if isinstance(entry, dict):
                cleaned.append(dict(entry))
        return cleaned

    def _normalize_term_groups_for_storage(self, groups: Iterable[dict]) -> List[dict]:
        normalized: List[dict] = []
        for entry in groups or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get('name', '')).strip()
            if not name:
                continue
            terms_raw = entry.get('terms') or []
            terms: List[str] = []
            seen_terms: Set[str] = set()
            for term in terms_raw:
                term_str = str(term).strip()
                if not term_str:
                    continue
                lower = term_str.lower()
                if lower in seen_terms:
                    continue
                seen_terms.add(lower)
                terms.append(term_str)
            if not terms:
                continue
            data: Dict[str, Any] = {'name': name, 'terms': terms}
            total_freq = entry.get('total_frequency')
            try:
                if total_freq is not None:
                    data['total_frequency'] = float(total_freq)
            except (TypeError, ValueError):
                pass
            missing_terms = entry.get('missing_terms')
            if isinstance(missing_terms, list):
                cleaned_missing = [str(term).strip() for term in missing_terms if str(term).strip()]
                if cleaned_missing:
                    data['missing_terms'] = cleaned_missing
            normalized.append(data)
        return normalized

    def _set_term_groups(self, groups: Iterable[dict], *, log_change: bool, show_notice: bool = False):
        parent = self.parent()
        if parent is None:
            return
        normalized = self._normalize_term_groups_for_storage(groups)
        previous = list(getattr(parent, 'collocation_term_groups', []))
        parent.collocation_term_groups = normalized
        state = dict(getattr(parent, 'collocation_state', {}) or {})
        state['term_groups'] = [dict(group) for group in normalized]
        parent.collocation_state = state
        self._update_group_summary()
        self._save_state()
        if log_change and normalized != previous:
            self._log_term_groups_change(previous, normalized)
        if show_notice:
            self._set_clear_notice('Run the collocation analysis again to apply changes.')

    def _log_term_groups_change(self, previous: List[dict], current: List[dict]):
        parent = self.parent()
        if parent is None:
            return
        if not current:
            parent.append_project_log('Text Analysis', ['<div>Term groups cleared.</div>'])
            return
        items = []
        for group in current:
            name = html.escape(str(group.get('name', '')))
            terms = group.get('terms', []) or []
            missing = {str(term).strip().lower() for term in group.get('missing_terms', []) or []}
            term_parts = []
            for term in terms:
                rendered = html.escape(str(term))
                if str(term).strip().lower() in missing:
                    rendered = f"{rendered} (not in list)"
                term_parts.append(rendered)
            freq = group.get('total_frequency')
            freq_text = ''
            if isinstance(freq, (float, int)):
                value = float(freq)
                if abs(value - round(value)) < 1e-6:
                    freq_text = f" (Total: {int(round(value)):,})"
                else:
                    freq_text = f" (Total: {value:.2f})"
            items.append(f'<li><strong>{name}</strong>: {"; ".join(term_parts)}{freq_text}</li>')
        lines = [
            f'<div>Updated term groups ({len(current)})</div>',
            f'<div style="max-height:220px; overflow-y:auto;"><ul>{"".join(items)}</ul></div>',
        ]
        parent.append_project_log('Text Analysis', lines)

    def _update_drop_summary(self):
        terms = self._get_parent_drop_terms()
        count = len(terms)
        if count:
            self.drop_summary_label.setText(f'Dropped terms: {count}')
            self.clear_drop_btn.setEnabled(True)
        else:
            self.drop_summary_label.setText('No terms dropped.')
            self.clear_drop_btn.setEnabled(False)
        if terms:
            body = '<br/>'.join(html.escape(term) for term in terms)
            self.drop_terms_view.setHtml(body)
        else:
            self.drop_terms_view.setHtml('<span style="color:#777777;">(none)</span>')
        self._set_section_badge(self._drop_section_button, count)
        if count > 0 and self._drop_terms_prev_count == 0:
            if not self.topic_remove_drop_tokens_check.isChecked():
                self.topic_remove_drop_tokens_check.blockSignals(True)
                self.topic_remove_drop_tokens_check.setChecked(True)
                self.topic_remove_drop_tokens_check.blockSignals(False)
        if count == 0:
            if self._drop_section_button is not None and self._drop_section_button.isChecked():
                self._drop_section_button.setChecked(False)
        elif self._drop_terms_prev_count == 0 and self._drop_section_button is not None and not self._drop_section_button.isChecked():
            self._drop_section_button.setChecked(True)
        self._drop_terms_prev_count = count
        self._update_topic_toggle_states()

    def _update_selected_terms_summary(self):
        terms = list(dict.fromkeys(self._rank_selected_terms))
        count = len(terms)
        if count:
            self.selected_summary_label.setText(f'Selected terms: {count}')
            body = '<br/>'.join(html.escape(term) for term in terms)
            self.selected_terms_view.setHtml(body)
            self.clear_selected_btn.setEnabled(True)
        else:
            self.selected_summary_label.setText('No terms selected.')
            self.selected_terms_view.setHtml('<span style="color:#777777;">(none)</span>')
            self.clear_selected_btn.setEnabled(False)
        self._set_section_badge(self._selected_section_button, count)
        if count == 0:
            if self._selected_section_button is not None and self._selected_section_button.isChecked():
                self._selected_section_button.setChecked(False)
        elif self._selected_terms_prev_count == 0 and self._selected_section_button is not None and not self._selected_section_button.isChecked():
            self._selected_section_button.setChecked(True)
        self._selected_terms_prev_count = count
        self._update_topic_toggle_states()

    def _update_group_summary(self):
        groups = self._get_parent_term_groups()
        count = len(groups)
        term_total = sum(len(group.get('terms', []) or []) for group in groups)
        if count:
            self.group_summary_label.setText(f'Term groups: {count} (covering {term_total} term(s))')
            self.clear_groups_btn.setEnabled(True)
        else:
            self.group_summary_label.setText('No term groups defined.')
            self.clear_groups_btn.setEnabled(False)
        if groups:
            lines = []
            for group in groups:
                name = html.escape(str(group.get('name', '')))
                terms = group.get('terms', []) or []
                missing = {str(term).strip().lower() for term in group.get('missing_terms', []) or []}
                term_parts = []
                for term in terms:
                    rendered = html.escape(str(term))
                    if str(term).strip().lower() in missing:
                        rendered = f"{rendered} <span style=\"color:#b45309;\">(not in list)</span>"
                    term_parts.append(rendered)
                terms_html = '; '.join(term_parts) if term_parts else '(none)'
                freq = group.get('total_frequency')
                freq_text = ''
                if isinstance(freq, (float, int)):
                    value = float(freq)
                    if abs(value - round(value)) < 1e-6:
                        freq_text = f" (Total: {int(round(value)):,})"
                    else:
                        freq_text = f" (Total: {value:.2f})"
                lines.append(f'<div><strong>{name}</strong>: {terms_html}{freq_text}</div>')
            body = ''.join(lines)
        else:
            body = '<span style="color:#777777;">(none)</span>'
        self.group_terms_view.setHtml(body)
        badge_count = term_total if term_total > 0 else count
        self._set_section_badge(self._group_section_button, badge_count)
        if count == 0:
            if self._group_section_button is not None and self._group_section_button.isChecked():
                self._group_section_button.setChecked(False)
        elif self._group_terms_prev_count == 0 and self._group_section_button is not None and not self._group_section_button.isChecked():
            self._group_section_button.setChecked(True)
        self._group_terms_prev_count = count

    def _parse_manual_terms(self, raw: str) -> List[str]:
        if not raw:
            return []
        terms: List[str] = []
        seen: Set[str] = set()
        for piece in re.split(r'[;,\r\n\t]+', raw):
            term = piece.strip()
            if not term:
                continue
            lowered = term.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            terms.append(term)
        return terms

    def _set_dropped_terms(self, terms: List[str], *, log_change: bool, show_notice: bool = False):
        parent = self.parent()
        if parent is None:
            return
        normalized: List[str] = []
        seen = set()
        for term in terms:
            term_str = str(term).strip()
            if term_str and term_str not in seen:
                seen.add(term_str)
                normalized.append(term_str)
        previous = list(getattr(parent, 'collocation_drop_terms', []))
        parent.collocation_drop_terms = normalized
        state = dict(getattr(parent, 'collocation_state', {}) or {})
        state['dropped_terms'] = list(normalized)
        parent.collocation_state = state
        self._update_drop_summary()
        self._save_state()
        if log_change and normalized != previous:
            self._log_drop_terms_change(previous, normalized)
        if show_notice:
            self._set_clear_notice('Run the collocation analysis again to apply changes.')

    def _log_drop_terms_change(self, previous: List[str], current: List[str]):
        parent = self.parent()
        if parent is None:
            return
        count = len(current)
        if count == 0:
            lines = ['<div>Dropped terms cleared.</div>']
        else:
            items = ''.join(f'<li>{html.escape(term)}</li>' for term in current)
            lines = [
                f'<div>Updated dropped terms ({count})</div>',
                f'<div style="max-height:220px; overflow-y:auto;"><ul>{items}</ul></div>',
            ]
        parent.append_project_log('Text Analysis', lines)

    def _set_clear_notice(self, text: str):
        if not hasattr(self, 'clear_notice_label'):
            return
        if text:
            self.clear_notice_label.setText(text)
            self.clear_notice_label.show()
        else:
            self.clear_notice_label.clear()
            self.clear_notice_label.hide()

    def add_custom_drop_terms(self):
        text, ok = QInputDialog.getMultiLineText(
            self,
            'Add Custom Drop Terms',
            'Enter drop terms (one per line or separated by commas):',
        )
        if not ok:
            return
        terms = self._parse_manual_terms(text)
        if not terms:
            return
        current = self._get_parent_drop_terms()
        seen = {term.lower() for term in current}
        updated = list(current)
        added = False
        for term in terms:
            lowered = term.lower()
            if lowered not in seen:
                seen.add(lowered)
                updated.append(term)
                added = True
        if not added:
            return
        self._set_dropped_terms(updated, log_change=True, show_notice=True)

    def clear_dropped_terms(self):
        if not self._get_parent_drop_terms():
            return
        self._set_dropped_terms([], log_change=True, show_notice=True)

    def add_custom_selected_terms(self):
        text, ok = QInputDialog.getMultiLineText(
            self,
            'Add Custom Selected Terms',
            'Enter selected terms (one per line or separated by commas):',
        )
        if not ok:
            return
        terms = self._parse_manual_terms(text)
        if not terms:
            return
        existing = list(dict.fromkeys(self._rank_selected_terms))
        seen = {term.lower() for term in existing}
        added = False
        for term in terms:
            lowered = term.lower()
            if lowered not in seen:
                existing.append(term)
                seen.add(lowered)
                added = True
        if not added:
            return
        self._rank_selected_terms = existing
        self._update_selected_terms_summary()
        self._save_state()

    def clear_selected_terms(self):
        if not self._rank_selected_terms:
            return
        self._rank_selected_terms = []
        self._update_selected_terms_summary()
        self._save_state()

    def clear_group_terms(self):
        if not self._get_parent_term_groups():
            return
        self._set_term_groups([], log_change=True, show_notice=True)

    def _update_select_terms_enabled(self):
        metrics_path = None
        if isinstance(self._last_output_paths, dict):
            metrics_path = self._last_output_paths.get('metrics')
        enabled = bool(metrics_path and os.path.exists(metrics_path))
        self.select_terms_btn.setEnabled(enabled)
        self.group_terms_btn.setEnabled(enabled)

    def _update_topic_documents_button(self):
        if not hasattr(self, 'btn_open_topic_docs'):
            return
        path = None
        if isinstance(self._last_topic_paths, dict):
            path = self._last_topic_paths.get('doc_topics')
        enabled = bool(path and os.path.exists(path))
        self.btn_open_topic_docs.setEnabled(enabled)


    def _fetch_metric_term_infos(self, metrics_path: str) -> List[dict]:
        try:
            df = pd.read_csv(metrics_path)
        except Exception as exc:
            raise RuntimeError(f'Unable to read metrics file: {exc}')
        if df.empty or 'collocate_term' not in df.columns:
            raise ValueError('No collocate terms are available.')
        df = df.dropna(subset=['collocate_term']).copy()
        if df.empty:
            raise ValueError('No collocate terms are available.')
        df['collocate_term'] = df['collocate_term'].astype(str)
        freq_col = 'frequency' if 'frequency' in df.columns else None
        if freq_col:
            df = df.sort_values([freq_col, 'collocate_term'], ascending=[False, True]).reset_index(drop=True)
        else:
            df = df.sort_values(['collocate_term']).reset_index(drop=True)
        term_infos: List[dict] = []
        for idx, row in df.iterrows():
            info = {
                'term': row['collocate_term'],
                'frequency': row.get(freq_col) if freq_col else None,
                'rank': idx + 1,
            }
            term_infos.append(info)
        return term_infos

    def open_drop_terms_dialog(self):
        term = self.term_input.text().strip()
        if not term:
            QMessageBox.warning(self, 'Search Term Required', 'Enter a search term before selecting terms to drop.')
            return
        city_text = self.city_combo.currentText()
        state_text = self.state_combo.currentText()
        city = None if not city_text or city_text == 'All Cities' else city_text.strip()
        state = None if not state_text or state_text == 'All States' else state_text.strip()
        paths = self._build_output_paths(
            term,
            self.start_input.text().strip(),
            self.end_input.text().strip(),
            city,
            state,
            self._options_with_context(),
        )
        metrics_path = paths.get('metrics')
        if not metrics_path or not os.path.exists(metrics_path):
            QMessageBox.warning(self, 'Metrics Not Found', 'Run the collocation analysis before selecting terms to drop.')
            return
        try:
            term_infos = self._fetch_metric_term_infos(metrics_path)
        except RuntimeError as exc:
            QMessageBox.critical(self, 'Read Error', str(exc))
            return
        except ValueError:
            QMessageBox.information(self, 'No Collocates', 'No collocated terms are available to drop.')
            return

        selected_set = set(self._get_parent_drop_terms())
        top_terms = term_infos[:150]
        known_terms = {info['term'] for info in top_terms}
        info_map = {info['term']: info for info in term_infos}

        # Include selected terms beyond the top 150 and any missing from the file
        for info in term_infos[150:]:
            term_name = info['term']
            if term_name in selected_set and term_name not in known_terms:
                top_terms.append(info)
                known_terms.add(term_name)
        for term_name in selected_set:
            if term_name not in known_terms:
                info = info_map.get(term_name)
                if info:
                    top_terms.append(info)
                else:
                    top_terms.append({'term': term_name, 'frequency': None, 'rank': None})
                known_terms.add(term_name)

        dialog = TermDropDialog(self, top_terms, selected_set)
        if dialog.exec_() == QDialog.Accepted:
            self._set_dropped_terms(dialog.selected_terms, log_change=True)

    def open_group_terms_dialog(self):
        if not self.group_terms_btn.isEnabled():
            return
        term = self.term_input.text().strip()
        if not term:
            QMessageBox.warning(self, 'Search Term Required', 'Enter a search term before grouping terms.')
            return
        city_text = self.city_combo.currentText()
        state_text = self.state_combo.currentText()
        city = None if not city_text or city_text == 'All Cities' else city_text.strip()
        state = None if not state_text or state_text == 'All States' else state_text.strip()
        paths = self._build_output_paths(
            term,
            self.start_input.text().strip(),
            self.end_input.text().strip(),
            city,
            state,
            self._options_with_context(),
        )
        metrics_path = paths.get('metrics')
        if not metrics_path or not os.path.exists(metrics_path):
            QMessageBox.warning(self, 'Metrics Not Found', 'Run the collocation analysis before grouping terms.')
            return
        try:
            term_infos = self._fetch_metric_term_infos(metrics_path)
        except RuntimeError as exc:
            QMessageBox.critical(self, 'Read Error', str(exc))
            return
        except ValueError:
            QMessageBox.information(self, 'No Collocates', 'No collocated terms are available to group.')
            return

        existing_groups = self._get_parent_term_groups()
        top_terms = term_infos[:150]
        known_terms = {info['term'] for info in top_terms}
        info_map = {info['term']: info for info in term_infos}

        for group in existing_groups:
            for term_name in group.get('terms', []) or []:
                if term_name in known_terms:
                    continue
                info = info_map.get(term_name)
                if info:
                    top_terms.append(info)
                else:
                    top_terms.append({'term': term_name, 'frequency': None, 'rank': None})
                known_terms.add(term_name)

        dialog = TermGroupDialog(self, top_terms, existing_groups)
        if dialog.exec_() == QDialog.Accepted:
            self._set_term_groups(dialog.created_groups, log_change=True, show_notice=True)
            # Return focus to Collocation dialog
            try:
                self.raise_()
                self.activateWindow()
                self.setFocus()
            except Exception:
                pass

    def open_select_terms_dialog(self):
        if not self.select_terms_btn.isEnabled():
            return
        term = self.term_input.text().strip()
        if not term:
            QMessageBox.warning(self, 'Search Term Required', 'Enter a search term before selecting terms.')
            return
        city_text = self.city_combo.currentText()
        state_text = self.state_combo.currentText()
        city = None if not city_text or city_text == 'All Cities' else city_text.strip()
        state = None if not state_text or state_text == 'All States' else state_text.strip()
        paths = self._build_output_paths(
            term,
            self.start_input.text().strip(),
            self.end_input.text().strip(),
            city,
            state,
            self._options_with_context(),
        )
        metrics_path = paths.get('metrics')
        if not metrics_path or not os.path.exists(metrics_path):
            QMessageBox.warning(self, 'Metrics Not Found', 'Run the collocation analysis before selecting terms.')
            return
        try:
            term_infos = self._fetch_metric_term_infos(metrics_path)
        except RuntimeError as exc:
            QMessageBox.critical(self, 'Read Error', str(exc))
            return
        except ValueError:
            QMessageBox.information(self, 'No Collocates', 'No collocated terms are available to select.')
            return

        selected_set = set(self._rank_selected_terms)
        top_terms = term_infos[:150]
        known_terms = {info['term'] for info in top_terms}
        info_map = {info['term']: info for info in term_infos}

        for info in term_infos[150:]:
            term_name = info['term']
            if term_name in selected_set and term_name not in known_terms:
                top_terms.append(info)
                known_terms.add(term_name)
        for term_name in selected_set:
            if term_name not in known_terms:
                info = info_map.get(term_name)
                if info:
                    top_terms.append(info)
                else:
                    top_terms.append({'term': term_name, 'frequency': None, 'rank': None})
                known_terms.add(term_name)

        dialog = TermPlotDialog(self, top_terms, selected_set)
        if dialog.exec_() == QDialog.Accepted:
            normalized: List[str] = []
            seen = set()
            for item in dialog.selected_terms:
                term_clean = str(item).strip()
                if term_clean and term_clean not in seen:
                    seen.add(term_clean)
                    normalized.append(term_clean)
            self._rank_selected_terms = normalized
            self._update_selected_terms_summary()
            self._save_state()

    def _current_time_bin_unit(self) -> Optional[str]:
        if self.ignore_bin.isChecked():
            return None
        size_text = self.bin_size.text().strip()
        if not size_text or not size_text.isdigit():
            return None
        return f"{int(size_text)} {self.bin_unit.currentText().lower()}"

    def _build_output_paths(self, term: str, start: Optional[str], end: Optional[str], city: Optional[str], state: Optional[str], options: dict):
        parent = self.parent()
        if parent is None:
            raise RuntimeError('Collocation dialog has no parent window')
        time_bin_unit = self._current_time_bin_unit()
        trimmed_options = dict(options)
        trimmed_options.pop('write_occurrences_geojson', None)
        return build_collocation_output_paths(
            parent.project_folder,
            term=term,
            start_date=start or None,
            end_date=end or None,
            city=city,
            state=state,
            time_bin_unit=time_bin_unit,
            ignore_bin=self.ignore_bin.isChecked(),
            options=trimmed_options,
            drop_terms=self._get_parent_drop_terms(),
            term_groups=self._get_parent_term_groups(),
        )

    def _register_preview(self, preview: QDialog):
        if preview is None:
            return
        self._preview_windows.append(preview)

        def _cleanup(_obj=None, ref=preview):
            if ref in self._preview_windows:
                self._preview_windows.remove(ref)

        preview.destroyed.connect(_cleanup)

    def _create_collapsible_section(
        self,
        title: str,
        content: QWidget,
        *,
        expanded: bool = False,
        on_toggle: Optional[Callable[[bool], None]] = None,
    ) -> Tuple[QWidget, QToolButton]:
        section = QWidget()
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(0)

        toggle = QToolButton(section)
        toggle._base_title = title  # type: ignore[attr-defined]
        self._set_section_badge(toggle, 0)
        toggle.setCheckable(True)
        toggle.setChecked(expanded)
        toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        toggle.setStyleSheet('QToolButton { border: none; font-weight: 600; padding: 2px 0; }')
        toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        section_layout.addWidget(toggle)

        body = QWidget(section)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(12, 6, 0, 0)
        body_layout.setSpacing(6)
        body_layout.addWidget(content)
        section_layout.addWidget(body)
        body.setVisible(expanded)

        def handle_toggle(checked: bool):
            toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
            body.setVisible(checked)
            if on_toggle:
                on_toggle(checked)

        toggle.toggled.connect(handle_toggle)
        return section, toggle

    def _set_section_badge(self, toggle: Optional[QToolButton], count: int):
        if toggle is None:
            return
        base_title = str(getattr(toggle, '_base_title', toggle.text()))
        if count > 0:
            toggle.setText(f'{base_title} ({count})')
        else:
            toggle.setText(base_title)

    def _confirm_overwrite_if_needed(self, paths: dict) -> bool:
        existing = [p for p in (paths.get('metrics'), paths.get('occurrences')) if p and os.path.exists(p)]
        if not existing:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle('Overwrite Existing Results?')
        box.setText('Collocation outputs already exist for these parameters.')
        box.setInformativeText('\n'.join(existing))
        overwrite_btn = box.addButton('Overwrite', QMessageBox.AcceptRole)
        open_btn = box.addButton('Open Existing', QMessageBox.ActionRole)
        cancel_btn = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(overwrite_btn)
        box.setEscapeButton(cancel_btn)
        box.exec_()
        clicked = box.clickedButton()
        if clicked == overwrite_btn:
            return True
        if clicked == open_btn:
            metrics = paths.get('metrics')
            if metrics and os.path.exists(metrics):
                preview = CSVPreviewDialog(metrics, parent=self, max_rows=150)
                preview.resize(1000, 620)
                preview.show()
                preview.raise_()
                preview.activateWindow()
                preview.setFocus()
                self._register_preview(preview)
            return False
        return False

    def _confirm_topic_overwrite_if_needed(self, paths: dict) -> bool:
        existing = [p for p in (paths.get('topics'), paths.get('doc_topics'), paths.get('by_time')) if p and os.path.exists(p)]
        if not existing:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle('Overwrite Existing Topic Model?')
        box.setText('Topic modeling outputs already exist for these parameters.')
        box.setInformativeText('\n'.join(existing))
        overwrite_btn = box.addButton('Overwrite', QMessageBox.AcceptRole)
        open_btn = box.addButton('Open Existing', QMessageBox.ActionRole)
        cancel_btn = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(overwrite_btn)
        box.setEscapeButton(cancel_btn)
        box.exec_()
        clicked = box.clickedButton()
        if clicked == overwrite_btn:
            return True
        if clicked == open_btn:
            topics_path = paths.get('topics')
            if topics_path and os.path.exists(topics_path):
                preview = CSVPreviewDialog(topics_path, parent=self, max_rows=150)
                preview.resize(1000, 620)
                preview.show()
                preview.raise_()
                preview.activateWindow()
                preview.setFocus()
                self._register_preview(preview)
            self._last_topic_paths = dict(paths)
            self._update_topic_documents_button()
            return False
        return False


    def _save_state(self):
        if self._loading_defaults:
            return
        parent = self.parent()
        if parent is None:
            return
        state = {
            'mode': 'geo' if self.mode_geo.isChecked() else 'json',
            'city': self.city_combo.currentText() if self.city_combo.currentIndex() > 0 else '',
            'state': self.state_combo.currentText() if self.state_combo.currentIndex() > 0 else '',
            'term': self.term_input.text().strip(),
            'start': self.start_input.text().strip(),
            'end': self.end_input.text().strip(),
            'bin_size': self.bin_size.text().strip(),
            'bin_unit': self.bin_unit.currentText(),
            'ignore_bin': self.ignore_bin.isChecked(),
            'options': self._collect_options(),
            'dropped_terms': self._get_parent_drop_terms(),
            'term_groups': self._get_parent_term_groups(),
            'context_left': self.context_left_spin.value(),
            'context_right': self.context_right_spin.value(),
            'rank_selected_terms': list(self._rank_selected_terms),
            'rank_log_scale': bool(getattr(self, '_rank_log_scale', True)),
            'topic_settings': self._collect_topic_settings(),
            'topic_trend_settings': dict(self._topic_trend_settings or {}),
        }
        parent.collocation_state = state

    def on_mode_toggle(self):
        if self.mode_geo.isChecked():
            # Enable city/state filters
            self.city_combo.setEnabled(True)
            self.state_combo.setEnabled(True)
            # Populate dropdowns if a GeoJSON is loaded
            if getattr(self.parent(), 'geojson_file', None) and os.path.exists(self.parent().geojson_file):
                self.populate_city_state()
            self._prefill_from_current_source()
        else:
            # Disable filters for JSON mode
            self.city_combo.setEnabled(False)
            self.state_combo.setEnabled(False)
            # Reset selections to "All"
            self.city_combo.setCurrentIndex(0)
            self.state_combo.setCurrentIndex(0)
            self._prefill_from_current_source()
        occ_cb = self.checks.get('write_occurrences_geojson')
        if occ_cb is not None:
            occ_cb.setEnabled(self.mode_geo.isChecked())
        if not self._loading_defaults:
            self._save_state()

    def populate_city_state(self):
        geo_path = getattr(self.parent(), 'geojson_file', None)
        if not geo_path or not os.path.exists(geo_path):
            return
        try:
            with open(geo_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            features = data.get('features', [])
        except Exception:
            return
        cities = sorted({feat['properties'].get('City') for feat in features if feat.get('properties') and feat['properties'].get('City')})
        states = sorted({feat['properties'].get('State') for feat in features if feat.get('properties') and feat['properties'].get('State')})
        cities = [c for c in cities if c not in [None, ""]]
        states = [s for s in states if s not in [None, ""]]
        # Populate city combo
        self.city_combo.blockSignals(True)
        self.city_combo.clear()
        self.city_combo.addItem('All Cities')
        for c in cities:
            self.city_combo.addItem(str(c))
        self.city_combo.blockSignals(False)
        # Populate state combo
        self.state_combo.blockSignals(True)
        self.state_combo.clear()
        self.state_combo.addItem('All States')
        for s in states:
            self.state_combo.addItem(str(s))
        self.state_combo.blockSignals(False)
        # Reset to "All" by default
        self.city_combo.setCurrentIndex(0)
        self.state_combo.setCurrentIndex(0)
        if self._loading_defaults:
            parent = self.parent()
            state = getattr(parent, 'collocation_state', {}) if parent else {}
            saved_city = state.get('city')
            if saved_city:
                idx = self.city_combo.findText(saved_city, Qt.MatchFixedString)
                if idx >= 0:
                    self.city_combo.setCurrentIndex(idx)
            saved_state = state.get('state')
            if saved_state:
                idx = self.state_combo.findText(saved_state, Qt.MatchFixedString)
                if idx >= 0:
                    self.state_combo.setCurrentIndex(idx)

    def run_collocate(self):
        # Gather input parameters
        city_text = self.city_combo.currentText()
        state_text = self.state_combo.currentText()
        city = None if not city_text or city_text == 'All Cities' else city_text.strip()
        state = None if not state_text or state_text == 'All States' else state_text.strip()
        start = self.start_input.text().strip()
        end   = self.end_input.text().strip()
        term  = self.term_input.text().strip()
        if not term:
            QMessageBox.warning(self, 'Search Term Required', 'Enter a search term before running collocation analysis.')
            return

        ignore_bin = self.ignore_bin.isChecked()
        if not ignore_bin:
            size_text = self.bin_size.text().strip()
            if not size_text.isdigit():
                QMessageBox.warning(self, 'Invalid Bin Size', 'Please enter an integer ≥ 1.')
                return

        time_bin_unit = self._current_time_bin_unit()
        write_by_time = False
        opts_bool = self._collect_options()
        window_left = self.context_left_spin.value()
        window_right = self.context_right_spin.value()
        options_with_context = dict(opts_bool)
        options_with_context['window_left'] = window_left
        options_with_context['window_right'] = window_right

        parent = self.parent()
        if parent is None:
            QMessageBox.warning(self, 'Unavailable', 'Parent window not available.')
            return
        metadata_enabled = getattr(parent, 'metadata_enabled', True)

        predicted_paths = self._build_output_paths(term, start or None, end or None, city, state, options_with_context)
        if not self._confirm_overwrite_if_needed(predicted_paths):
            return

        if self.mode_json.isChecked():
            json_path = getattr(parent, 'json_file', None)
            if not json_path or not os.path.exists(json_path):
                self.choose_source_file()
                json_path = getattr(parent, 'json_file', None)
                if not json_path or not os.path.exists(json_path):
                    return
            source_kwargs = {'json_path': json_path, 'geojson_path': None}
        else:
            geo_path = getattr(parent, 'geojson_file', None)
            if not geo_path or not os.path.exists(geo_path):
                self.choose_source_file()
                geo_path = getattr(parent, 'geojson_file', None)
                if not geo_path or not os.path.exists(geo_path):
                    return
            source_kwargs = {'json_path': None, 'geojson_path': geo_path}

        def task(*, cancel_event: Optional[threading.Event]):
            return run_collocation(
                parent.project_folder,
                city=city,
                state=state,
                start_date=start or None,
                end_date=end or None,
                term=term,
                time_bin_unit=time_bin_unit,
                ignore_bin=ignore_bin,
                write_by_time=write_by_time,
                drop_terms=parent.collocation_drop_terms,
                term_groups=getattr(parent, 'collocation_term_groups', []),
                window_left=window_left,
                window_right=window_right,
                metadata_enabled=metadata_enabled,
                cancel_event=cancel_event,
                **source_kwargs,
                **opts_bool,
            )

        def handle_success(result: Any):
            if not isinstance(result, dict):
                QMessageBox.information(self, 'Collocation', 'Collocation completed with no outputs.')
                return
            metrics_path = result.get('metrics')
            if metrics_path and os.path.exists(metrics_path):
                self.raise_()
                self.activateWindow()
                preview = CSVPreviewDialog(metrics_path, parent=self, max_rows=150)
                preview.resize(1000, 620)
                preview.show()
                preview.raise_()
                preview.activateWindow()
                preview.setFocus()
                self._register_preview(preview)

            self._last_output_paths = result
            self._update_select_terms_enabled()
            self._save_state()
            mode_label = 'GeoJSON' if self.mode_geo.isChecked() else 'JSON'
            self._log_collocation_run(
                mode_label,
                term,
                start or 'all',
                end or 'all',
                city or 'All',
                state or 'All',
                time_bin_unit,
                ignore_bin,
                options_with_context,
                result,
            )
            self._set_clear_notice('')

        def handle_error(exc: Exception):
            QMessageBox.critical(self, 'Error', str(exc))

        def handle_cancel():
            self._set_clear_notice('Collocation analysis cancelled.')

        self._start_operation(
            'Running collocation analysis…',
            task,
            on_success=handle_success,
            on_error=handle_error,
            on_cancel=handle_cancel,
            context='collocation',
        )

    def run_topic_model_action(self):
        term = self.term_input.text().strip()
        if not term:
            QMessageBox.warning(self, 'Search Term Required', 'Enter a search term before running topic modeling.')
            return

        city_text = self.city_combo.currentText()
        state_text = self.state_combo.currentText()
        city = None if not city_text or city_text == 'All Cities' else city_text.strip()
        state = None if not state_text or state_text == 'All States' else state_text.strip()
        start = self.start_input.text().strip()
        end = self.end_input.text().strip()

        parent = self.parent()
        if parent is None:
            QMessageBox.warning(self, 'Unavailable', 'Parent window not available.')
            return

        drop_terms = self._get_parent_drop_terms()
        selected_terms = list(dict.fromkeys(self._rank_selected_terms))
        topic_settings = self._collect_topic_settings()
        params = self._topic_parameters(drop_terms, selected_terms, settings=topic_settings)
        include_urls = bool(topic_settings.get('include_article_url'))
        effective_selected = selected_terms if params.restrict_to_selected_terms else []

        predicted_paths = self._build_topic_output_paths_topic(
            term,
            start or None,
            end or None,
            city,
            state,
            params,
            effective_selected,
        )
        if not self._confirm_topic_overwrite_if_needed(predicted_paths):
            return
        self._last_topic_paths = dict(predicted_paths)
        self._update_topic_documents_button()

        json_path = getattr(parent, 'json_file', None)
        geo_path = getattr(parent, 'geojson_file', None)

        if self.mode_json.isChecked():
            if not json_path or not os.path.exists(json_path):
                self.choose_source_file()
                json_path = getattr(parent, 'json_file', None)
                if not json_path or not os.path.exists(json_path):
                    return
            source_kwargs = {'json_path': json_path, 'geojson_path': None}
        else:
            if not geo_path or not os.path.exists(geo_path):
                self.choose_source_file()
                geo_path = getattr(parent, 'geojson_file', None)
                if not geo_path or not os.path.exists(geo_path):
                    return
            source_kwargs = {'json_path': None, 'geojson_path': geo_path}

        time_bin_unit = self._current_time_bin_unit()
        metadata_enabled = getattr(parent, 'metadata_enabled', True)
        window_left = self.context_left_spin.value()
        window_right = self.context_right_spin.value()

        def task(*, cancel_event: Optional[threading.Event]):
            return run_topic_model(
                parent.project_folder,
                term=term,
                city=city,
                state=state,
                start_date=start or None,
                end_date=end or None,
                time_bin_unit=time_bin_unit,
                ignore_bin=self.ignore_bin.isChecked(),
                params=params,
                drop_terms=drop_terms,
                term_groups=self._get_parent_term_groups(),
                selected_terms=effective_selected,
                include_article_url=include_urls,
                metadata_enabled=metadata_enabled,
                cancel_event=cancel_event,
                window_left=window_left,
                window_right=window_right,
                **source_kwargs,
            )

        def handle_success(result: Any):
            if not isinstance(result, dict):
                QMessageBox.information(self, 'Topic Modeling', 'Topic modeling completed with no outputs.')
                return
            self._last_topic_paths = result
            self._update_topic_documents_button()
            topics_path = result.get('topics')
            if topics_path and os.path.exists(topics_path):
                preview = CSVPreviewDialog(topics_path, parent=self, max_rows=150)
                preview.resize(1000, 620)
                preview.show()
                preview.raise_()
                preview.activateWindow()
                preview.setFocus()
                self._register_preview(preview)

            start_display = start or 'all'
            end_display = end or 'all'
            city_display = city or 'All'
            state_display = state or 'All'
            self._log_topic_model_run(term, start_display, end_display, city_display, state_display, params, result, include_urls)
            self._set_clear_notice('')
            self._save_state()

        def handle_error(exc: Exception):
            QMessageBox.critical(self, 'Topic Modeling Error', str(exc))
            self._update_topic_documents_button()

        def handle_cancel():
            self._set_clear_notice('Topic modeling cancelled.')
            self._update_topic_documents_button()

        self._start_operation(
            'Running topic modeling…',
            task,
            on_success=handle_success,
            on_error=handle_error,
            on_cancel=handle_cancel,
            context='topic-model',
            cancel_requested_message='Topic modeling cancellation requested. The current iteration may take additional time to finish.',
        )

    def open_topic_documents_csv(self):
        path = None
        if isinstance(self._last_topic_paths, dict):
            path = self._last_topic_paths.get('doc_topics')
        if not path or not os.path.exists(path):
            QMessageBox.information(self, 'File Not Found', 'Run topic modeling to create the topic_documents CSV before opening it.')
            return
        preview = CSVPreviewDialog(path, parent=self, max_rows=150)
        preview.resize(1000, 620)
        preview.show()
        preview.raise_()
        preview.activateWindow()
        preview.setFocus()
        self._register_preview(preview)


    def show_bar(self):
        term = self.term_input.text().strip()
        if not term:
            QMessageBox.warning(self, 'Search Term Required', 'Enter a search term to view bar charts.')
            return
        city_text = self.city_combo.currentText()
        state_text = self.state_combo.currentText()
        city = None if not city_text or city_text == 'All Cities' else city_text.strip()
        state = None if not state_text or state_text == 'All States' else state_text.strip()
        start_value = self.start_input.text().strip()
        end_value = self.end_input.text().strip()
        options_with_context = self._options_with_context()
        paths = self._build_output_paths(term, start_value, end_value, city, state, options_with_context)
        metrics_path = paths.get('metrics')
        if not metrics_path or not os.path.exists(metrics_path):
            QMessageBox.warning(self, 'File Not Found', 'Metrics file not found. Please run the collocation analysis first.')
            return
        selected_terms = list(dict.fromkeys(self._rank_selected_terms))
        try:
            plot_bar = _import_plot_bar()
            if selected_terms:
                try:
                    df = pd.read_csv(metrics_path)
                except Exception as exc:
                    QMessageBox.critical(self, 'Read Error', f'Unable to read metrics file:\n{exc}')
                    return
                if df.empty or 'collocate_term' not in df.columns or 'frequency' not in df.columns:
                    QMessageBox.information(
                        self,
                        'No Data',
                        'Selected terms could not be displayed because the metrics file is missing required data.',
                    )
                    selected_terms = []
                else:
                    df = df.dropna(subset=['collocate_term']).copy()
                    df['collocate_term'] = df['collocate_term'].astype(str)
                    filtered = df[df['collocate_term'].isin(selected_terms)].copy()
                    available = set(filtered['collocate_term'])
                    missing = [term for term in selected_terms if term not in available]
                    if filtered.empty:
                        QMessageBox.information(
                            self,
                            'Terms Not Found',
                            'None of the selected terms are present in the metrics file. Showing default top terms instead.',
                        )
                        selected_terms = []
                    else:
                        if missing:
                            preview = ', '.join(missing[:5])
                            if len(missing) > 5:
                                preview += ', …'
                            QMessageBox.information(
                                self,
                                'Terms Not Found',
                                'The following selected terms are not present in the metrics file and were skipped:\n' + preview,
                            )
                        fig = plot_bar(filtered)
                        if fig is not None:
                            fig.canvas.mpl_connect('close_event', lambda event: self._refocus_collocation())
                        return
            fig = plot_bar(metrics_path)
        except Exception as exc:
            QMessageBox.critical(self, 'Plot Error', str(exc))
            return
        if fig is not None:
            fig.canvas.mpl_connect('close_event', lambda event: self._refocus_collocation())

    def _display_rank_chart_from_path(
        self,
        file_path: str,
        *,
        settings: dict,
        log_scale: bool,
        drop_terms_set: Set[str],
        term: str,
        city_text: str,
        state_text: str,
        start_value: str,
        end_value: str,
        time_bin_unit: Optional[str],
    ) -> None:
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            QMessageBox.critical(self, 'Error', f'Could not read by-time data:\n{exc}')
            return
        self._render_rank_chart_dataframe(
            df,
            file_path=file_path,
            settings=settings,
            log_scale=log_scale,
            drop_terms_set=drop_terms_set,
            term=term,
            city_text=city_text,
            state_text=state_text,
            start_value=start_value,
            end_value=end_value,
            time_bin_unit=time_bin_unit,
        )

    def _render_rank_chart_dataframe(
        self,
        df: pd.DataFrame,
        *,
        file_path: Optional[str],
        settings: dict,
        log_scale: bool,
        drop_terms_set: Set[str],
        term: str,
        city_text: str,
        state_text: str,
        start_value: str,
        end_value: str,
        time_bin_unit: Optional[str],
    ) -> None:
        if df is None or df.empty:
            QMessageBox.information(self, 'No Rank Data', 'No collocate rank data available for the selected parameters.')
            return
        df_local = df.copy()
        if drop_terms_set:
            df_local = df_local[~df_local['collocate_term'].isin(drop_terms_set)].reset_index(drop=True)
        required_cols = {'time_bin', 'collocate_term', 'ordinal_rank'}
        if not required_cols.issubset(df_local.columns):
            QMessageBox.information(self, 'No Rank Data', 'No collocate rank data available for the selected parameters.')
            return
        if df_local.empty:
            QMessageBox.information(self, 'No Rank Data', 'No collocate rank data available for the selected parameters.')
            return
        try:
            bins_ordered = sorted(
                df_local['time_bin'].dropna().unique(),
                key=lambda x: pd.to_datetime(str(x), errors='coerce'),
            )
        except Exception:
            bins_ordered = sorted(df_local['time_bin'].dropna().unique())
        if not bins_ordered:
            bins_ordered = self._derive_time_bins_from_inputs()
            if not bins_ordered:
                QMessageBox.information(self, 'No Rank Data', 'No collocate rank data available for the selected parameters.')
                return
        unique_terms = df_local['collocate_term'].dropna().astype(str).unique().tolist()
        if not unique_terms:
            QMessageBox.information(self, 'No Rank Data', 'No collocate terms available to plot.')
            return

        top_n = settings.get('top_n', 10)
        use_global = bool(settings.get('use_global'))
        show_labels = bool(settings.get('show_labels'))
        home_idx = int(settings.get('home_bin_index', 0))
        home_idx = max(0, min(home_idx, len(bins_ordered) - 1))
        use_selected_terms = bool(settings.get('use_selected_terms'))

        unique_term_set = set(unique_terms)
        manual_terms = [term for term in self._rank_selected_terms if term in unique_term_set]
        missing_manual = [term for term in self._rank_selected_terms if term not in unique_term_set]
        manual_mode = use_selected_terms and bool(manual_terms)
        if manual_mode and missing_manual:
            missing_preview = ', '.join(missing_manual[:5])
            if len(missing_manual) > 5:
                missing_preview += ', …'
            QMessageBox.information(
                self,
                'Terms Not Found',
                'The following selected terms are not available for the current filters and will be skipped:\n' + missing_preview,
            )
        elif use_selected_terms and not manual_mode and self._rank_selected_terms:
            QMessageBox.information(
                self,
                'Terms Not Found',
                'None of the manually selected terms are available for the current filters. The chart will use the Top N terms instead.',
            )

        home_label_value = bins_ordered[home_idx] if bins_ordered else None

        if manual_mode:
            top_terms = manual_terms
            legend_order = manual_terms
        elif use_global:
            if 'frequency' in df_local.columns:
                try:
                    freq_series = df_local.groupby('collocate_term')['frequency'].sum(min_count=1)
                except TypeError:
                    freq_series = df_local.groupby('collocate_term')['frequency'].sum()
            else:
                freq_series = None
            if freq_series is None or freq_series.empty:
                freq_series = df_local.groupby('collocate_term').size()
            if freq_series.empty:
                QMessageBox.information(self, 'No Data', 'No data available for the selected terms.')
                return
            summary = freq_series.reset_index()
            if summary.shape[1] >= 2:
                summary.columns = ['collocate_term', 'total_frequency'] + list(summary.columns[2:])
            else:
                summary.columns = ['collocate_term', 'total_frequency']
            summary['term_key'] = summary['collocate_term'].astype(str).str.lower()
            summary = summary.sort_values(['total_frequency', 'term_key'], ascending=[False, True])
            top_terms = summary.head(top_n)['collocate_term'].tolist()
            if not top_terms:
                QMessageBox.information(self, 'No Data', 'No data available for the selected terms.')
                return
            legend_order = top_terms
        else:
            df_home = df_local[df_local['time_bin'] == home_label_value].dropna(subset=['ordinal_rank'])
            if df_home.empty:
                QMessageBox.information(self, 'No Data', 'The selected bin contains no collocates.')
                return
            if 'frequency' in df_home.columns:
                freq_home = df_home.groupby('collocate_term')['frequency'].sum()
            else:
                freq_home = df_home.groupby('collocate_term').size()
            freq_home = freq_home.sort_values(ascending=False)
            top_terms = freq_home.head(top_n).index.tolist()
            if not top_terms:
                QMessageBox.information(self, 'No Data', 'No data available for the selected terms.')
                return
            legend_order = top_terms

        if manual_mode and not top_terms:
            QMessageBox.information(self, 'No Data', 'None of the selected terms are available for the current filters.')
            return

        df_top = df_local[df_local['collocate_term'].isin(top_terms)].copy()
        if df_top.empty:
            QMessageBox.information(self, 'No Data', 'No data available for the selected terms.')
            return

        if use_global:
            home_label_display = 'All time (global)'
        else:
            try:
                home_label_display = str(home_label_value)
            except Exception:
                home_label_display = 'Selected bin'

        city_display = city_text or 'All Cities'
        state_display = state_text or 'All States'
        start_display = start_value or 'All dates'
        end_display = end_value or 'All dates'
        if self.ignore_bin.isChecked() or not time_bin_unit:
            time_unit_display = 'Time Bin: none'
        else:
            time_unit_display = f"Time Bin: {time_bin_unit}"

        settings_text = self._build_rank_chart_summary(
            term_count=len(top_terms),
            home_label=home_label_display,
            city_label=city_display,
            state_label=state_display,
            start_label=start_display,
            end_label=end_display,
            time_unit=time_unit_display,
            keyword=term,
        )

        try:
            plot_rank_changes = _import_plot_rank_changes()
            fig = plot_rank_changes(
                df_top,
                legend_order=legend_order,
                show_term_labels=show_labels,
                settings_text=settings_text,
                use_log_scale=log_scale,
            )
        except Exception as exc:
            QMessageBox.critical(self, 'Plot Error', str(exc))
            return

        if fig is not None:
            fig.canvas.mpl_connect('close_event', lambda event: self._refocus_collocation())
            self._set_clear_notice('')
    def show_topic_trends(self):
        if self.ignore_bin.isChecked():
            QMessageBox.information(self, 'Time Bins Required', 'Enable time binning to plot topic trends.')
            return
        term = self.term_input.text().strip()
        if not term:
            QMessageBox.warning(self, 'Search Term Required', 'Enter a search term before plotting topic trends.')
            return
        time_bin_unit = self._current_time_bin_unit()
        if not time_bin_unit:
            QMessageBox.warning(self, 'Invalid Bin Size', 'Enter a valid bin size and time unit to plot topic trends.')
            return
        start_value = self.start_input.text().strip()
        end_value = self.end_input.text().strip()
        city_text = self.city_combo.currentText()
        state_text = self.state_combo.currentText()
        city = None if not city_text or city_text == 'All Cities' else city_text.strip()
        state = None if not state_text or state_text == 'All States' else state_text.strip()

        drop_terms = self._get_parent_drop_terms()
        selected_terms = list(dict.fromkeys(self._rank_selected_terms))
        topic_settings = self._collect_topic_settings()
        params = self._topic_parameters(drop_terms, selected_terms, settings=topic_settings)
        effective_selected = selected_terms if params.restrict_to_selected_terms else []

        target_path = None
        if isinstance(self._last_topic_paths, dict):
            guess = self._last_topic_paths.get('by_time')
            if guess and os.path.exists(guess):
                target_path = guess
        if not target_path:
            paths = self._build_topic_output_paths_topic(
                term,
                start_value or None,
                end_value or None,
                city,
                state,
                params,
                effective_selected,
            )
            candidate = paths.get('by_time')
            if candidate and os.path.exists(candidate):
                target_path = candidate

        if not target_path:
            QMessageBox.information(self, 'Data Not Found', 'Run the topic model with time binning to generate trends first.')
            return

        defaults = dict(self._topic_trend_settings or {})
        max_topics_available = max(1, params.n_topics)
        dialog = TopicTrendsSettingsDialog(self, defaults=defaults, max_topics=max_topics_available)
        if dialog.exec_() != QDialog.Accepted:
            return
        trend_settings = dialog.values()
        top_topics = int(trend_settings.get('top_topics') or max_topics_available)
        top_topics = max(1, min(top_topics, max_topics_available))
        metric_choice = str(trend_settings.get('metric') or 'weight_sum')
        legend_topics = bool(trend_settings.get('legend_topics', True))
        label_points = bool(trend_settings.get('label_points', True))
        log_scale = bool(trend_settings.get('log_scale'))
        self._topic_trend_settings = {
            'top_topics': top_topics,
            'metric': metric_choice,
            'legend_topics': legend_topics,
            'label_points': label_points,
            'log_scale': log_scale,
        }
        self._save_state()

        metric_messages = {
            'weight_sum': 'Topic Weight',
            'ordinal_rank': 'Average Topic Rank',
            'doc_count': 'Article Count',
        }
        metric_label = metric_messages.get(metric_choice, metric_choice)
        city_display = city_text.strip() if city_text and city_text.strip() and city_text.strip() != 'All Cities' else 'All Cities'
        state_display = state_text.strip() if state_text and state_text.strip() and state_text.strip() != 'All States' else 'All States'
        start_display = start_value or 'All dates'
        end_display = end_value or 'All dates'
        time_unit_display = f"Time Bin: {time_bin_unit}"
        settings_text = self._build_topic_trend_summary(
            keyword=term,
            model_label=params.model.upper(),
            trained_topics=params.n_topics,
            top_topics=top_topics,
            metric_label=metric_label,
            city_label=city_display,
            state_label=state_display,
            start_label=start_display,
            end_label=end_display,
            time_unit=time_unit_display,
        )

        try:
            plot_topics = _import_plot_topics_over_time()
            fig = plot_topics(
                target_path,
                top_n=top_topics,
                metric=metric_choice,
                show_legend=legend_topics,
                label_points=label_points,
                log_scale=log_scale,
                settings_text=settings_text,
            )
        except Exception as exc:
            QMessageBox.critical(self, 'Plot Error', str(exc))
            return
        if fig is not None:
            fig.canvas.mpl_connect('close_event', lambda event: self._refocus_collocation())

    def _log_topic_model_run(
        self,
        term: str,
        start: str,
        end: str,
        city: str,
        state: str,
        params: TopicModelParameters,
        paths: Optional[dict],
        include_urls: bool,
    ):
        parent = self.parent()
        if parent is None or not hasattr(parent, 'append_project_log'):
            return

        options = []
        if params.drop_stopwords:
            options.append('drop stopwords')
        if params.exclude_drop_term_documents:
            options.append('exclude drop terms')
        if params.remove_drop_terms_from_tokens:
            options.append('remove drop tokens')
        if params.restrict_to_selected_terms:
            options.append('selected-term filter')
        if params.max_documents:
            options.append(f'max docs {params.max_documents}')
        if include_urls:
            options.append('include article URLs')

        lines = [
            f'<div>Term: {html.escape(term or "(none)")}</div>',
            f'<div>Dates: {html.escape(start)} → {html.escape(end)}</div>',
            f'<div>Location: {html.escape(city)}, {html.escape(state)}</div>',
            f'<div>Model: {html.escape(params.model.upper())} | Topics: {params.n_topics} | Top words: {params.n_top_words}</div>',
            f'<div>Max features: {params.max_features} | Options: {html.escape(", ".join(options) if options else "none")}</div>',
        ]

        if params.restrict_to_selected_terms:
            lines.append(f'<div>Selected terms used: {len(self._rank_selected_terms)} term(s)</div>')
        drop_terms = self._get_parent_drop_terms()
        if drop_terms:
            lines.append(f'<div>Drop terms active: {len(drop_terms)} term(s)</div>')

        if isinstance(paths, dict):
            topics_path = paths.get('topics')
            doc_topics_path = paths.get('doc_topics')
            by_time_path = paths.get('by_time')
            if topics_path and os.path.exists(topics_path):
                encoded = urllib.parse.quote(topics_path)
                lines.append(f'<div>Topics CSV: <a href="chronam-open:{encoded}">{html.escape(topics_path)}</a></div>')
            if doc_topics_path and os.path.exists(doc_topics_path):
                encoded = urllib.parse.quote(doc_topics_path)
                lines.append(f'<div>Article-topic CSV: <a href="chronam-open:{encoded}">{html.escape(doc_topics_path)}</a></div>')
            if by_time_path and os.path.exists(by_time_path):
                encoded = urllib.parse.quote(by_time_path)
                lines.append(f'<div>Topics-by-time CSV: <a href="chronam-open:{encoded}">{html.escape(by_time_path)}</a></div>')

        parent.append_project_log('Topic Modeling', lines)

    def _log_collocation_run(self, mode: str, term: str, start: str, end: str, city: str, state: str,
                              time_bin_unit: Optional[str], ignore_bin: bool, options: dict,
                              paths: dict):
        parent = self.parent()
        if parent is None:
            return
        summary_parts = [
            f"Source: {mode}",
            f"Term: {term or '(none)'}",
            f"Dates: {start} → {end}",
            f"City: {city}",
            f"State: {state}",
        ]
        if ignore_bin:
            summary_parts.append('Time bin: ignored')
        else:
            summary_parts.append(f"Time bin: {time_bin_unit or 'default'}")

        context_left = options.get('window_left')
        context_right = options.get('window_right')
        if context_left is not None or context_right is not None:
            try:
                left_val = int(context_left) if context_left is not None else 0
            except (TypeError, ValueError):
                left_val = 0
            try:
                right_val = int(context_right) if context_right is not None else 0
            except (TypeError, ValueError):
                right_val = 0
            summary_parts.append(f"Context: {left_val} left / {right_val} right")

        enabled_opts = [name for name, enabled in options.items() if isinstance(enabled, bool) and enabled]
        summary_parts.append(f"Options: {', '.join(enabled_opts) if enabled_opts else 'none'}")
        drop_count = len(getattr(parent, 'collocation_drop_terms', []))
        if drop_count:
            summary_parts.append(f"Dropped terms: {drop_count}")
        group_count = len(getattr(parent, 'collocation_term_groups', []))
        if group_count:
            summary_parts.append(f"Term groups: {group_count}")
        lines = [f"<div>{html.escape('; '.join(summary_parts))}</div>"]

        def link_line(label: str, path: Optional[str]):
            if not path:
                return None
            encoded = urllib.parse.quote(path)
            return f'<div>{html.escape(label)}: <a href="chronam-open:{encoded}">{html.escape(path)}</a></div>'

        metrics_line = link_line('Metrics CSV', paths.get('metrics'))
        if metrics_line:
            lines.append(metrics_line)
        by_time_path = paths.get('by_time')
        if by_time_path:
            lines.append(link_line('By-time CSV', by_time_path))
        elif ignore_bin:
            lines.append('<div>By-time CSV not generated (ignore bin size enabled).</div>')
        occ_line = link_line('Occurrences GeoJSON', paths.get('occurrences'))
        if occ_line:
            lines.append(occ_line)
        if drop_count:
            terms = getattr(parent, 'collocation_drop_terms', [])
            if terms:
                items = ''.join(f'<li>{html.escape(term)}</li>' for term in terms)
                lines.append(
                    f'<div><strong>Dropped terms list:</strong></div>'
                    f'<div style="max-height:220px; overflow-y:auto;"><ul>{items}</ul></div>'
                )
        if group_count:
            groups = getattr(parent, 'collocation_term_groups', []) or []
            if groups:
                group_items = []
                for group in groups:
                    name = html.escape(str(group.get('name', '')))
                    terms = group.get('terms', []) or []
                    missing = {str(term).strip().lower() for term in group.get('missing_terms', []) or []}
                    term_parts = []
                    for term in terms:
                        rendered = html.escape(str(term))
                        if str(term).strip().lower() in missing:
                            rendered = f"{rendered} (not in list)"
                        term_parts.append(rendered)
                    freq = group.get('total_frequency')
                    freq_text = ''
                    if isinstance(freq, (float, int)):
                        value = float(freq)
                        if abs(value - round(value)) < 1e-6:
                            freq_text = f" (Total: {int(round(value)):,})"
                        else:
                            freq_text = f" (Total: {value:.2f})"
                    group_items.append(f'<li><strong>{name}</strong>: {"; ".join(term_parts)}{freq_text}</li>')
                lines.append(
                    '<div><strong>Term groups:</strong></div>'
                    f'<div style="max-height:220px; overflow-y:auto;"><ul>{"".join(group_items)}</ul></div>'
                )

        parent.append_project_log('Text Analysis', lines)

    def closeEvent(self, event):
        self._save_state()
        super().closeEvent(event)

    def _refocus_collocation(self):
        self.raise_()
        self.activateWindow()

    def show_rank(self):
        term = self.term_input.text().strip()
        if not term:
            QMessageBox.warning(self, 'Search Term Required', 'Enter a search term to view rank changes.')
            return
        city_text = self.city_combo.currentText()
        state_text = self.state_combo.currentText()
        city = None if not city_text or city_text == 'All Cities' else city_text.strip()
        state = None if not state_text or state_text == 'All States' else state_text.strip()
        start_value = self.start_input.text().strip()
        end_value = self.end_input.text().strip()
        options_with_context = self._options_with_context()
        paths = self._build_output_paths(term, start_value, end_value, city, state, options_with_context)
        file_path = paths.get('by_time')
        time_bin_unit = self._current_time_bin_unit()
        if not file_path:
            QMessageBox.warning(
                self,
                'Time Bin Required',
                'Enable time bin settings before viewing rank changes.',
            )
            return

        parent = self.parent()
        metadata_enabled = getattr(parent, 'metadata_enabled', True) if parent is not None else True
        opts_bool = self._collect_options()
        window_left = self.context_left_spin.value()
        window_right = self.context_right_spin.value()
        drop_terms_raw = self._get_parent_drop_terms()
        drop_terms = [str(t).strip() for t in drop_terms_raw if str(t).strip()]
        drop_terms_set = set(drop_terms)
        prefer_geo = self.mode_geo.isChecked()

        csv_exists = bool(file_path and os.path.exists(file_path))
        df: Optional[pd.DataFrame] = None
        bins_ordered: List[Any] = []
        unique_terms: List[str] = []

        if csv_exists:
            try:
                df = pd.read_csv(file_path)
            except Exception:
                df = None
                csv_exists = False

        if df is not None:
            required_cols = {'time_bin', 'collocate_term', 'ordinal_rank'}
            if not required_cols.issubset(df.columns):
                df = None
                csv_exists = False
            else:
                if drop_terms_set:
                    df = df[~df['collocate_term'].isin(drop_terms_set)].reset_index(drop=True)
                if not df.empty:
                    try:
                        bins_ordered = sorted(
                            df['time_bin'].unique(),
                            key=lambda x: pd.to_datetime(str(x), errors='coerce'),
                        )
                    except Exception:
                        bins_ordered = sorted(df['time_bin'].unique())
                    unique_terms = df['collocate_term'].dropna().unique().tolist()
                else:
                    df = None
                    csv_exists = False

        if drop_terms_set:
            csv_status = 'drop_terms'
        elif csv_exists:
            csv_status = 'existing'
        else:
            csv_status = 'missing'

        selected_manual = list(dict.fromkeys(self._rank_selected_terms))
        max_terms_dialog = len(unique_terms) if unique_terms else 150
        default_top = min(10, max_terms_dialog) if max_terms_dialog else 1
        if not bins_ordered:
            bins_ordered = self._derive_time_bins_from_inputs()
        settings_dialog = CollocationRankSettingsDialog(
            self,
            bins_ordered,
            max(1, max_terms_dialog),
            default_top,
            selected_terms=selected_manual,
            csv_status=csv_status,
            csv_path=file_path,
            drop_terms=drop_terms,
            log_scale=getattr(self, '_rank_log_scale', True),
        )
        if settings_dialog.exec_() != QDialog.Accepted:
            return
        settings = settings_dialog.values()
        self._rank_selected_terms = list(dict.fromkeys(settings.get('selected_terms', [])))
        self._update_selected_terms_summary()
        log_scale = bool(settings.get('log_scale', True))
        self._rank_log_scale = log_scale
        self._save_state()
        regen_on_accept = bool(drop_terms_set) or not csv_exists or df is None

        def render_from_path(path: str) -> None:
            self._display_rank_chart_from_path(
                path,
                settings=settings,
                log_scale=log_scale,
                drop_terms_set=drop_terms_set,
                term=term,
                city_text=city_text,
                state_text=state_text,
                start_value=start_value,
                end_value=end_value,
                time_bin_unit=time_bin_unit,
            )

        if regen_on_accept:
            def task(*, cancel_event: Optional[threading.Event]):
                return self._generate_by_time_csv(
                    city=city,
                    state=state,
                    start_value=start_value or None,
                    end_value=end_value or None,
                    term=term,
                    time_bin_unit=time_bin_unit,
                    drop_terms=drop_terms,
                    options_runtime=opts_bool,
                    options_hash=options_with_context,
                    window_left=window_left,
                    window_right=window_right,
                    metadata_enabled=metadata_enabled,
                    prefer_geo=prefer_geo,
                    cancel_event=cancel_event,
                )

            def handle_success(result: Tuple[Optional[str], Optional[str]]):
                built_path, error = result
                if not built_path:
                    QMessageBox.critical(self, 'Error', error or 'Unable to build by-time data.')
                    return
                render_from_path(built_path)

            def handle_error(exc: Exception):
                QMessageBox.critical(self, 'Error', str(exc))

            def handle_cancel():
                self._set_clear_notice('Rank changes build cancelled.')

            started = self._start_operation(
                'Building rank changes…',
                task,
                on_success=handle_success,
                on_error=handle_error,
                on_cancel=handle_cancel,
                context='rank-changes',
            )
            if started:
                return
            return
        else:
            if df is not None:
                self._render_rank_chart_dataframe(
                    df,
                    file_path=file_path,
                    settings=settings,
                    log_scale=log_scale,
                    drop_terms_set=drop_terms_set,
                    term=term,
                    city_text=city_text,
                    state_text=state_text,
                    start_value=start_value,
                    end_value=end_value,
                    time_bin_unit=time_bin_unit,
                )
                return
            if file_path and os.path.exists(file_path):
                render_from_path(file_path)
                return
            QMessageBox.information(self, 'By-time Not Found', 'Run the collocation analysis with time bins before viewing rank changes.')

    def _build_rank_chart_summary(
        self,
        *,
        term_count: int,
        home_label: str,
        city_label: str,
        state_label: str,
        start_label: str,
        end_label: str,
        time_unit: str,
        keyword: str,
    ) -> str:
        keyword_display = keyword or '(none)'
        line1 = f"Terms plotted: {term_count} | Home bin: {home_label} | Keyword: {keyword_display}"
        line2 = f"City filter: {city_label} | State filter: {state_label}"
        line3 = f"Dates: {start_label} – {end_label} | {time_unit or 'Time Bin: n/a'}"
        return '\n'.join([line1, line2, line3])

    def _build_topic_trend_summary(
        self,
        *,
        keyword: str,
        model_label: str,
        trained_topics: int,
        top_topics: int,
        metric_label: str,
        city_label: str,
        state_label: str,
        start_label: str,
        end_label: str,
        time_unit: str,
    ) -> str:
        keyword_display = keyword or '(none)'
        parts = [
            f"Keyword: {keyword_display} | Model: {model_label} | Topics trained: {trained_topics}",
            f"Top topics plotted: {top_topics} | Metric: {metric_label}",
            f"City filter: {city_label} | State filter: {state_label}",
            f"Dates: {start_label} – {end_label} | {time_unit}",
        ]
        return '\n'.join(parts)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
