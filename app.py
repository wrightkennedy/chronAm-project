import sys
import os
import json
import time
import subprocess
import os
import pandas as pd
import re
import threading
import html
import urllib.parse
from typing import Optional, List
from datetime import datetime, date
from PyQt5.QtWidgets import (
    QFileDialog, QApplication, QMainWindow, QAction, QWidget,
    QApplication, QMainWindow, QAction, QFileDialog, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QProgressBar, QMessageBox, QCheckBox, QFormLayout, QDialog, QTextBrowser, QComboBox,
    QTableWidget, QTableWidgetItem, QRadioButton, QButtonGroup, QDockWidget, QDialogButtonBox, QSpinBox,
    QDoubleSpinBox, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, pyqtProperty, QUrl, QObject, QEvent
from PyQt5.QtGui import QPainter, QPen, QIntValidator, QTextCursor, QKeySequence

from chronam import download_data
from chronam.map_create import create_map
from chronam.merge import merge_geojson
from chronam.collocate import run_collocation, build_collocation_output_paths
from chronam.visualize import plot_bar, plot_rank_changes


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
    progress = pyqtSignal(int)
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
        self.collocation_state = {}
        self.map_settings = {
            'mode': 'points',
            'time_unit': 'week',
            'time_step': 1,
            'linger_unit': 'week',
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
        self.btn_collocate = QPushButton('C) Run Collocation Analysis')
        self.btn_collocate.clicked.connect(self.action_collocate)
        self.btn_map = QPushButton('D) Create Map')
        self.btn_map.clicked.connect(self.open_create_map_dialog)
        for btn in (self.btn_download, self.btn_update, self.btn_collocate, self.btn_map):
            main_layout.addWidget(btn)

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
        self.collocation_state = {}
        self.map_settings = {
            'mode': 'points',
            'time_unit': 'week',
            'time_step': 1,
            'linger_unit': 'week',
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
        }

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
        self.collocation_state = {}
        self.map_settings = {
            'mode': 'points',
            'time_unit': 'week',
            'time_step': 1,
            'linger_unit': 'week',
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
        }

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
        start_dir = os.path.join(self.project_folder, 'data', 'raw') if self.project_folder else ''
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
        cleaning_layout.addWidget(self.clean_lowercase_cb)
        cleaning_layout.addWidget(self.clean_urls_cb)
        cleaning_layout.addWidget(self.clean_hyphen_cb)
        layout.addWidget(cleaning_group)

        self.log = QTextBrowser()
        self.log.setOpenLinks(False)
        self.log.anchorClicked.connect(self._handle_log_link)
        self.log.setReadOnly(True)
        self.progress = QProgressBar()
        layout.addWidget(self.log)
        layout.addWidget(self.progress)

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

        self._current_run_lines = []
        dataset_folder = self.parent().ensure_dataset_folder()
        if not dataset_folder:
            self._log_plain('Search cancelled — dataset folder not recognized.')
            self._finalize_project_log()
            return

        self.current_parquet_dir = dataset_folder
        self.refresh_dataset_label()

        # Single output path for the full range
        out_path = os.path.join(
            self.parent().project_folder, 'data', 'raw',
            f"{term}_{start}_{end}.json"
        )
        if os.path.exists(out_path):
            if QMessageBox.warning(
                self, 'Overwrite Warning', f'Will overwrite:\n{out_path}',
                QMessageBox.Yes | QMessageBox.No
            ) != QMessageBox.Yes:
                self._log_plain('Search cancelled — existing file retained.')
                self._finalize_project_log()
                return

        if self.log.toPlainText().strip():
            self._log_blank()

        header = f'Searching for "{term}" between {start} and {end}'
        self._log_plain(header)
        self._log_plain('Starting search...')
        self.progress.setValue(0)
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
            }
        )
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.download_finished)
        self.thread.start()

    def update_progress(self, count: int):
        now = time.time()
        elapsed = now - self._start_time
        year_value = None
        year_elapsed = None

        def parquet_dir_candidates():
            explicit_dir = getattr(self, 'current_parquet_dir', None)
            if explicit_dir:
                yield explicit_dir

            override_dir = getattr(self, 'current_raw_folder', None)
            if override_dir:
                yield override_dir

            project_dir = getattr(self.parent(), 'project_folder', None)
            if project_dir:
                yield os.path.join(project_dir, 'data', 'parquet')
                yield os.path.join(project_dir, 'parquet')
                dataset_from_parent = getattr(self.parent(), 'dataset_folder', None)
                if dataset_from_parent:
                    yield dataset_from_parent
                yield project_dir

        def parse_year(text):
            match = re.match(r"(\d{4})", text or "")
            return int(match.group(1)) if match else None

        start_year = parse_year(self.start_input.text().strip())
        end_year = parse_year(self.end_input.text().strip())
        year_pattern = re.compile(r"AmericanStories_(\d{4})\.parquet")

        year_range = []
        if start_year and end_year:
            lo, hi = sorted((start_year, end_year))
            year_range = [str(y) for y in range(lo, hi + 1)]

        seen_dirs = set()

        for directory in parquet_dir_candidates():
            if not directory or directory in seen_dirs:
                continue
            seen_dirs.add(directory)
            if not os.path.isdir(directory):
                continue

            try:
                files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
            except OSError:
                continue

            year_map = {}
            for file in files:
                match = year_pattern.fullmatch(file)
                if match:
                    year_map[match.group(1)] = file

            for target_year in year_range or sorted(year_map.keys()):
                if target_year in year_map and target_year not in self.logged_years:
                    self.logged_years.add(target_year)
                    year_value = target_year
                    if self._year_timer is None:
                        self._year_timer = now
                    year_elapsed = max(0.0, now - self._year_timer)
                    self._year_timer = now
                    break

            if year_value is not None:
                break

        self.progress.setValue(count)
        if year_value is not None:
            if year_elapsed is None:
                year_elapsed = max(0.0, now - (self._year_timer or now))
            self._log_plain(
                f"Found {count:,} articles in year {year_value} - search time {year_elapsed:.1f}s"
            )
        else:
            self._log_plain(f"Found {count:,} articles — elapsed {elapsed:.1f}s")



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
        if isinstance(result, Exception):
            QMessageBox.critical(self, 'Error', str(result))
            self._log_plain(f"Search failed: {result}")
            self._cancel_event.clear()
            self._cancel_requested = False
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
            self._finalize_project_log()
            return

        self._log_separator()

        # Automatically load the last downloaded JSON
        last_json = result[-1] if result else None
        if last_json:
            p = self.parent()
            p.json_file = last_json
            p._update_loaded_file_labels()

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
        parent = self.parent()
        candidates = []
        explicit = getattr(parent, 'locations_csv_path', None)
        if explicit:
            candidates.append(explicit)
        dataset = getattr(parent, 'dataset_folder', None)
        if dataset:
            candidates.append(os.path.join(os.path.dirname(dataset), 'ChronAm_newspapers_XY.csv'))
        candidates.append(os.path.join(parent.project_folder, 'data', 'ChronAm_newspapers_XY.csv'))
        for cand in candidates:
            if cand and os.path.exists(cand):
                return cand
        return candidates[0] if candidates else None

    def prompt_csv(self, show_hint: bool = False) -> Optional[str]:
        parent = self.parent()
        if show_hint and not self._csv_hint_shown:
            QMessageBox.information(
                self,
                'Locate Locations CSV',
                'Locate the newspaper locations table named "ChronAm_newspapers_XY.csv".'
            )
            self._csv_hint_shown = True
        start = parent.project_folder if parent else os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Select Locations CSV (ChronAm_newspapers_XY.csv)',
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

        try:
            out_paths = merge_geojson(
                parent.project_folder,
                csv_path=self.csv_path,
                search_term=term,
                year=year,
                json_path=self.json_path
            )
            if out_paths:
                last_geo = out_paths[-1]
                parent.geojson_file = last_geo
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
        lines = []
        total_articles = 0
        places_all = set()
        for path in out_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    geo = json.load(f)
                features = geo.get('features', [])
                total_articles += len(features)
                places = set()
                for feat in features:
                    props = feat.get('properties', {})
                    places.add((props.get('Title'), props.get('SN')))
                places_all.update(places)
                encoded = urllib.parse.quote(path)
                lines.append(
                    f'<div>Output GeoJSON: <a href="chronam-open:{encoded}">{html.escape(path)}</a></div>'
                )
            except Exception:
                continue

        if not lines:
            lines.append('<div>No GeoJSON files created.</div>')

        summary = f'<div>Joined {len(places_all):,} places to {total_articles:,} articles.</div>'
        parent.append_project_log('Add Geographic Info', [summary] + lines)

class CSVPreviewDialog(QDialog):
    def __init__(self, csv_path, parent=None, max_rows=100):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(csv_path))
        self.setMinimumSize(900, 600)
        df = pd.read_csv(csv_path).head(max_rows)
        tbl = QTableWidget(df.shape[0], df.shape[1], self)
        tbl.setHorizontalHeaderLabels(list(df.columns))
        for i, row in df.iterrows():
            for j, val in enumerate(row):
                tbl.setItem(i, j, QTableWidgetItem(str(val)))
        layout = QVBoxLayout(self)
        layout.addWidget(tbl)
        tbl.setFocus()
        self.table = tbl


class CollocationRankSettingsDialog(QDialog):
    def __init__(self, parent, bins: List[str], max_terms: int, default_top_n: int = 10):
        super().__init__(parent)
        self.setWindowTitle('Rank Chart Settings')
        layout = QVBoxLayout(self)

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
        form.addRow(self.labels_check)

        layout.addLayout(form)

        self.global_check.toggled.connect(self.home_combo.setDisabled)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict:
        return {
            'top_n': self.top_spin.value(),
            'home_bin_index': self.home_combo.currentIndex(),
            'use_global': self.global_check.isChecked(),
            'show_labels': self.labels_check.isChecked(),
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
        idx = self.time_unit.findText(str(defaults.get('time_unit', 'week')), Qt.MatchFixedString)
        if idx >= 0:
            self.time_unit.setCurrentIndex(idx)

        self.disable_time.setChecked(bool(defaults.get('disable_time', False)))

        self.linger_step.setValue(max(0, int(defaults.get('linger_step', 2))))
        idx = self.linger_unit.findText(str(defaults.get('linger_unit', 'week')), Qt.MatchFixedString)
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

class CollocationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Collocation Analysis')
        self.setMinimumSize(620, 580)
        layout = QVBoxLayout(self)
        self._last_output_paths = None
        self._preview_windows = []

        # --- Source selection & status line ---
        mode_row = QHBoxLayout()
        self.mode_geo = QRadioButton('Use GeoJSON')
        self.mode_json = QRadioButton('Use JSON results')
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.mode_geo)
        self.mode_group.addButton(self.mode_json)

        # Default selection based on loaded files
        if getattr(parent, 'geojson_file', None):
            self.mode_geo.setChecked(True)
        elif getattr(parent, 'json_file', None):
            self.mode_json.setChecked(True)
        else:
            self.mode_geo.setChecked(True)

        mode_row.addWidget(self.mode_geo)
        mode_row.addWidget(self.mode_json)

        self.choose_btn = QPushButton('Choose File…')
        mode_row.addWidget(self.choose_btn)
        layout.addLayout(mode_row)

        self.source_label = QLabel(self._source_text())
        self.source_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.source_label)

        # Connect mode toggles and file chooser
        self.mode_geo.toggled.connect(lambda _: self.source_label.setText(self._source_text()))
        self.mode_json.toggled.connect(lambda _: self.source_label.setText(self._source_text()))
        self.mode_geo.toggled.connect(self.on_mode_toggle)
        self.mode_json.toggled.connect(self.on_mode_toggle)
        self.choose_btn.clicked.connect(self.choose_source_file)

        # 2) Parameter form layout
        form = QFormLayout()

        # City filter (dropdown)
        self.city_combo = QComboBox()
        self.city_combo.addItem('All Cities')
        city_row = QWidget(); city_layout = QHBoxLayout(city_row)
        city_layout.setContentsMargins(0, 0, 0, 0)
        city_layout.addWidget(self.city_combo)
        form.addRow('City:', city_row)

        # State filter (dropdown)
        self.state_combo = QComboBox()
        self.state_combo.addItem('All States')
        state_row = QWidget(); state_layout = QHBoxLayout(state_row)
        state_layout.setContentsMargins(0, 0, 0, 0)
        state_layout.addWidget(self.state_combo)
        form.addRow('State:', state_row)

        # Date range inputs
        self.start_input = QLineEdit()
        self.end_input = QLineEdit()
        form.addRow('Start Date:', self.start_input)
        form.addRow('End Date:', self.end_input)

        # Search term input
        self.term_input = QLineEdit()
        form.addRow('Search Term:', self.term_input)

        # Time bin controls
        self.bin_size = QLineEdit('1')
        self.bin_size.setValidator(QIntValidator(1, 1000, self))
        self.bin_size.setMaximumWidth(60)
        form.addRow('Bin Size:', self.bin_size)
        self.bin_unit = QComboBox()
        self.bin_unit.addItems(['Days', 'Weeks', 'Months', 'Years'])
        form.addRow('Time Unit:', self.bin_unit)
        self.ignore_bin = QCheckBox('Ignore Bin Size (no time binning)')
        self.ignore_bin.setChecked(True)
        form.addRow(self.ignore_bin)

        # Additional collocation options (checkboxes)
        self.checks = {}
        for opt in ['include_page_count', 'include_first_last_date', 'include_cooccurrence_rate', 'include_relative_position', 'drop_stopwords']:
            cb = QCheckBox(opt)
            cb.setChecked(True)
            form.addRow(cb)
            self.checks[opt] = cb

        layout.addLayout(form)

        self._loading_defaults = True
        self._restore_state_or_defaults()
        self._loading_defaults = False

        self.bin_size.textEdited.connect(self._handle_bin_control_change)
        self.bin_unit.currentIndexChanged.connect(self._handle_bin_control_change)
        self.ignore_bin.stateChanged.connect(lambda *_: self._save_state())
        self.city_combo.currentIndexChanged.connect(lambda *_: self._save_state())
        self.state_combo.currentIndexChanged.connect(lambda *_: self._save_state())
        self.term_input.textEdited.connect(lambda _text: self._save_state())
        self.start_input.textEdited.connect(lambda _text: self._save_state())
        self.end_input.textEdited.connect(lambda _text: self._save_state())
        self.mode_geo.toggled.connect(lambda _: self._save_state())
        self.mode_json.toggled.connect(lambda _: self._save_state())
        for cb in self.checks.values():
            cb.stateChanged.connect(lambda *_: self._save_state())

        # Action buttons
        btn_run = QPushButton('Run Collocation')
        btn_bar = QPushButton('Show Bar Chart')
        btn_rank = QPushButton('Show Rank Changes')
        btn_run.clicked.connect(self.run_collocate)
        btn_bar.clicked.connect(self.show_bar)
        btn_rank.clicked.connect(self.show_rank)
        for b in (btn_run, btn_bar, btn_rank):
            layout.addWidget(b)

    def _source_text(self):
        if self.mode_geo.isChecked():
            p = getattr(self.parent(), 'geojson_file', None)
            return f"GeoJSON: {os.path.basename(p) if p else '<none selected>'}"
        else:
            p = getattr(self.parent(), 'json_file', None)
            return f"JSON: {os.path.basename(p) if p else '<none selected>'}"

    def choose_source_file(self):
        parent = self.parent()
        if self.mode_geo.isChecked():
            p, _ = QFileDialog.getOpenFileName(self, 'Select GeoJSON File', parent.project_folder, 'GeoJSON Files (*.geojson *.json)')
            if p:
                parent.geojson_file = p
                parent._update_loaded_file_labels()
                # Update city/state lists for new GeoJSON
                self.populate_city_state()
        else:
            p, _ = QFileDialog.getOpenFileName(self, 'Select JSON Results', parent.project_folder, 'JSON Files (*.json)')
            if p:
                parent.json_file = p
                parent._update_loaded_file_labels()
        # Update source label text
        self.source_label.setText(self._source_text())
        if parent is not None:
            parent.collocation_state = {}
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
            self.on_mode_toggle()
            self._prefill_from_current_source()

    def _apply_state(self, state: dict):
        parent = self.parent()
        mode = state.get('mode', 'geo')
        self.mode_geo.blockSignals(True)
        self.mode_json.blockSignals(True)
        if mode == 'json' and getattr(parent, 'json_file', None):
            self.mode_json.setChecked(True)
        elif mode == 'geo' and getattr(parent, 'geojson_file', None):
            self.mode_geo.setChecked(True)
        elif getattr(parent, 'json_file', None):
            self.mode_json.setChecked(True)
        else:
            self.mode_geo.setChecked(True)
        self.mode_geo.blockSignals(False)
        self.mode_json.blockSignals(False)
        self.on_mode_toggle()

        # Prefill term/date from current source if missing in saved state
        self._prefill_from_current_source()

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
            cb.setChecked(bool(opts.get(key, True)))

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
        if meta.get('start_date'):
            self.start_input.setText(meta.get('start_date'))
        if meta.get('end_date'):
            self.end_input.setText(meta.get('end_date'))

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
        return build_collocation_output_paths(
            parent.project_folder,
            term=term,
            start_date=start or None,
            end_date=end or None,
            city=city,
            state=state,
            time_bin_unit=time_bin_unit,
            ignore_bin=self.ignore_bin.isChecked(),
            options=options,
        )

    def _register_preview(self, preview: QDialog):
        if preview is None:
            return
        self._preview_windows.append(preview)

        def _cleanup(_obj=None, ref=preview):
            if ref in self._preview_windows:
                self._preview_windows.remove(ref)

        preview.destroyed.connect(_cleanup)

    def _confirm_overwrite_if_needed(self, paths: dict) -> bool:
        existing = [p for p in (paths.get('metrics'), paths.get('by_time'), paths.get('occurrences')) if p and os.path.exists(p)]
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
        write_by_time = not ignore_bin
        opts = self._collect_options()

        parent = self.parent()
        if parent is None:
            QMessageBox.warning(self, 'Unavailable', 'Parent window not available.')
            return

        predicted_paths = self._build_output_paths(term, start or None, end or None, city, state, opts)
        if not self._confirm_overwrite_if_needed(predicted_paths):
            return

        try:
            if self.mode_json.isChecked():
                json_path = getattr(parent, 'json_file', None)
                if not json_path or not os.path.exists(json_path):
                    self.choose_source_file()
                    json_path = getattr(parent, 'json_file', None)
                    if not json_path:
                        return
                result = run_collocation(
                    parent.project_folder,
                    city=city,
                    state=state,
                    start_date=start or None,
                    end_date=end or None,
                    term=term,
                    time_bin_unit=time_bin_unit,
                    json_path=json_path,
                    geojson_path=None,
                    write_occurrences_geojson=False,
                    ignore_bin=ignore_bin,
                    write_by_time=write_by_time,
                    **opts,
                )
            else:
                geo_path = getattr(parent, 'geojson_file', None)
                if not geo_path or not os.path.exists(geo_path):
                    self.choose_source_file()
                    geo_path = getattr(parent, 'geojson_file', None)
                    if not geo_path:
                        return
                result = run_collocation(
                    parent.project_folder,
                    city=city,
                    state=state,
                    start_date=start or None,
                    end_date=end or None,
                    term=term,
                    time_bin_unit=time_bin_unit,
                    geojson_path=geo_path,
                    json_path=None,
                    write_occurrences_geojson=True,
                    ignore_bin=ignore_bin,
                    write_by_time=write_by_time,
                    **opts,
                )
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
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
            opts,
            result,
        )

    def show_bar(self):
        term = self.term_input.text().strip()
        if not term:
            QMessageBox.warning(self, 'Search Term Required', 'Enter a search term to view bar charts.')
            return
        city_text = self.city_combo.currentText()
        state_text = self.state_combo.currentText()
        city = None if not city_text or city_text == 'All Cities' else city_text.strip()
        state = None if not state_text or state_text == 'All States' else state_text.strip()
        paths = self._build_output_paths(term, self.start_input.text().strip(), self.end_input.text().strip(), city, state, self._collect_options())
        metrics_path = paths.get('metrics')
        if not metrics_path or not os.path.exists(metrics_path):
            QMessageBox.warning(self, 'File Not Found', 'Metrics file not found. Please run the collocation analysis first.')
            return
        fig = plot_bar(metrics_path)
        if fig is not None:
            fig.canvas.mpl_connect('close_event', lambda event: self._refocus_collocation())

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
        enabled_opts = [name for name, enabled in options.items() if enabled]
        summary_parts.append(f"Options: {', '.join(enabled_opts) if enabled_opts else 'none'}")
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

        parent.append_project_log('Collocation Analysis', lines)

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
        paths = self._build_output_paths(term, self.start_input.text().strip(), self.end_input.text().strip(), city, state, self._collect_options())
        file_path = paths.get('by_time')
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, 'No Data', 'Collocation by-time data not found. Run collocation with a time bin first.')
            return
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Could not read by-time data: {e}')
            return
        if df.empty or 'time_bin' not in df.columns or 'collocate_term' not in df.columns or 'ordinal_rank' not in df.columns:
            QMessageBox.information(self, 'No Rank Data', 'No collocate rank data available for the selected parameters.')
            return
        try:
            bins_ordered = sorted(df['time_bin'].unique(), key=lambda x: pd.to_datetime(str(x), errors='coerce'))
        except Exception:
            bins_ordered = sorted(df['time_bin'].unique())
        if not bins_ordered:
            QMessageBox.information(self, 'No Rank Data', 'No collocate rank data available.')
            return
        unique_terms = df['collocate_term'].dropna().unique().tolist()
        if not unique_terms:
            QMessageBox.information(self, 'No Rank Data', 'No collocate terms available to plot.')
            return
        default_top = min(10, len(unique_terms)) or 1
        settings_dialog = CollocationRankSettingsDialog(self, bins_ordered, len(unique_terms), default_top)
        if settings_dialog.exec_() != QDialog.Accepted:
            return
        settings = settings_dialog.values()
        top_n = settings['top_n']
        use_global = settings['use_global']
        show_labels = settings['show_labels']

        averages = None
        if use_global:
            averages = df.groupby('collocate_term')['ordinal_rank'].mean().dropna()
            top_terms = averages.sort_values().head(top_n).index.tolist()
        else:
            home_idx = settings['home_bin_index']
            home_idx = max(0, min(home_idx, len(bins_ordered) - 1))
            home_label = bins_ordered[home_idx]
            df_home = df[df['time_bin'] == home_label].dropna(subset=['ordinal_rank'])
            if df_home.empty:
                QMessageBox.information(self, 'No Data', 'The selected bin contains no collocates.')
                return
            df_home_sorted = df_home.sort_values('ordinal_rank')
            top_terms = df_home_sorted.head(top_n)['collocate_term'].tolist()

        if not top_terms:
            QMessageBox.information(self, 'No Data', 'No data available for the selected terms.')
            return

        df_top = df[df['collocate_term'].isin(top_terms)].copy()
        if df_top.empty:
            QMessageBox.information(self, 'No Data', 'No data available for the selected terms.')
            return

        if use_global:
            if averages is None:
                averages = df_top.groupby('collocate_term')['ordinal_rank'].mean().dropna()
            ordered_series = averages.reindex(top_terms).dropna().sort_values()
            legend_order = ordered_series.index.tolist()
            if not legend_order:
                legend_order = top_terms
        else:
            legend_order = top_terms

        fig = plot_rank_changes(df_top, legend_order=legend_order, show_term_labels=show_labels)
        fig.canvas.mpl_connect('close_event', lambda event: self._refocus_collocation())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
