import sys
import os
import json
import time
import subprocess
import os
import pandas as pd
import re
from datetime import datetime, date
from PyQt5.QtWidgets import (
    QFileDialog, QApplication, QMainWindow, QAction, QWidget,
    QApplication, QMainWindow, QAction, QFileDialog, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QProgressBar, QMessageBox, QListWidget, QListWidgetItem,
    QCheckBox, QFormLayout, QInputDialog, QDialog, QTextEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, pyqtProperty
from PyQt5.QtGui import QPainter, QPen, QIntValidator

from chronam import download_data
from chronam.map_create import create_map
from chronam.merge import merge_geojson
from chronam.fetch_metadata import fetch_missing_metadata
from chronam.collocate import run_collocation
from chronam.visualize import plot_bar, plot_rank_changes

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChronAM GUI")
        self.resize(500, 300)
        self.project_folder = os.getcwd()
        self.json_file = None
        self.geojson_file = None
        self.init_ui()

    def init_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        file_menu.addAction(self._action('New Project', self.new_project))
        file_menu.addAction(self._action('Open Project', self.open_project))
        file_menu.addSeparator()

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

        btn_download = QPushButton('A) Search Dataset')
        btn_download.clicked.connect(self.action_download)
        btn_update = QPushButton('B) Add Geographic Info')
        btn_update.clicked.connect(self.action_update_locations)
        btn_collocate = QPushButton('C) Run Collocation Analysis')
        btn_collocate.clicked.connect(self.action_collocate)
        btn_map = QPushButton('D) Create Map')
        btn_map.clicked.connect(self.action_create_map)
        for btn in (btn_download, btn_update, btn_collocate, btn_map):
            main_layout.addWidget(btn)

    def _action(self, name, slot):
        a = QAction(name, self)
        a.triggered.connect(slot)
        return a

    def new_project(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder to Create Project')
        if folder:
            name, ok = QInputDialog.getText(self, 'Project Name', 'Enter project folder name:')
            if ok and name:
                path = os.path.join(folder, name)
                os.makedirs(os.path.join(path, 'data', 'raw'), exist_ok=True)
                os.makedirs(os.path.join(path, 'data', 'processed'), exist_ok=True)
                self.project_folder = path
                QMessageBox.information(self, 'Project Created', f'Project at: {path}')

    def open_project(self):
        folder = QFileDialog.getExistingDirectory(self, 'Open Project Folder')
        if folder:
            self.project_folder = folder
            QMessageBox.information(self, 'Project Opened', f'Project at: {folder}')

    def open_json_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Load JSON File', '', 'JSON Files (*.json)')
        if file_path:
            self.json_file = file_path
            self.json_label.setText(os.path.basename(file_path))

    def open_geojson_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Load GeoJSON File', '', 'GeoJSON Files (*.geojson *.json)')
        if file_path:
            self.geojson_file = file_path
            self.geojson_label.setText(os.path.basename(file_path))

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


    def action_create_map(self):
        """Create an interactive map from the currently loaded GeoJSON."""
        # Ensure we have a geojson file selected
        if not self.geojson_file or not os.path.exists(self.geojson_file):
            self.open_geojson_file()
            if not self.geojson_file or not os.path.exists(self.geojson_file):
                return

        # Ask for map mode
        modes = ['Points', 'Heatmap']
        mode, ok = QInputDialog.getItem(
            self,
            'Map Mode',
            'Select map mode:',
            modes,
            0,
            False
        )
        if not ok:
            return
        # Example: ask for unit, step, linger
        units = ["day", "week", "month", "year"]
        time_unit, ok = QInputDialog.getItem(self, "Time Unit", "Unit:", units, 1, False)
        if not ok: return
        time_step, ok = QInputDialog.getInt(self, "Time Step", "Step:", 1, 1, 24)
        if not ok: return
        linger_unit, ok = QInputDialog.getItem(self, "Linger Unit", "Unit:", units, 1, False)
        if not ok: return
        linger_step, ok = QInputDialog.getInt(self, "Linger After", "Length:", 2, 0, 52)
        if not ok: return
        
        try:
            out_html = create_map(
                self.geojson_file,
                mode=str(mode).strip().lower(),
                time_unit=time_unit,
                time_step=time_step,
                linger_unit=linger_unit,
                linger_step=linger_step
            )
#         try:
#             # out_html = create_map(self.geojson_file, mode=str(mode).strip().lower())
#             out_html = create_map(
#                 self.geojson_file,
#                 mode=str(mode).strip().lower(),                
#                 time_unit="week",
#                 time_step=1,
#                 linger_unit="week",
#                 linger_step=2,   # points remain 2 weeks after their date
# )
        except Exception as e:
            QMessageBox.critical(self, 'Map Error', f'Failed to create map:\n{e}')
            return

        # Open in browser
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(out_html))
        QMessageBox.information(self, 'Map Created', f'Interactive map saved to:\n{out_html}')

class DownloadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Search Dataset')
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout(self)

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

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.progress = QProgressBar()
        layout.addWidget(self.log)
        layout.addWidget(self.progress)

        btns = QHBoxLayout()
        self.run_btn = QPushButton('Run Download')
        self.run_btn.clicked.connect(self.start_download)
        self.cancel_btn = QPushButton('Cancel Download')
        self.cancel_btn.clicked.connect(self.cancel_download)
        btns.addWidget(self.run_btn)
        btns.addWidget(self.cancel_btn)
        layout.addLayout(btns)

        self.thread = None
        self._start_time = None

    def start_download(self):
        self.parent().setEnabled(False)
        self.spinner.show()
        self._anim.start()

        term  = self.search_input.text().strip()
        start = self.start_input.text().strip()
        end   = self.end_input.text().strip()

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
                # User chose not to overwrite
                self.spinner.hide()
                self._anim.stop()
                self.parent().setEnabled(True)
                return

        self.log.append('Starting search...')
        self.progress.setValue(0)
        self._start_time = time.time()

        # Launch download in a separate thread
        self.thread = WorkerThread(download_data, self.parent().project_folder, term, start, end)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.download_finished)
        self.thread.start()

    def update_progress(self, count: int):
        elapsed = time.time() - self._start_time
        self.progress.setValue(count)
        self.log.append(f"Found {count:,} articles — elapsed {elapsed:.1f}s")

    def cancel_download(self):
        if self.thread and self.thread.isRunning():
            if QMessageBox.question(self, 'Cancel Download', 'Cancel the download?', QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                self.thread.terminate()
                self._anim.stop()
                self.spinner.hide()
                self.parent().setEnabled(True)
                self.log.append("Download cancelled.")

    def download_finished(self, result):
        elapsed = time.time() - self._start_time
        self._anim.stop()
        self.spinner.hide()
        self.parent().setEnabled(True)
        if isinstance(result, Exception):
            QMessageBox.critical(self, 'Error', str(result))
        else:
            QMessageBox.information(
                self, 'Download Complete',
                f'Finished in {elapsed:.1f}s\n' + '\n'.join(result)
            )
            # Automatically load the last downloaded JSON
            last_json = result[-1] if result else None
            if last_json:
                p = self.parent()
                p.json_file = last_json
                p.json_label.setText(os.path.basename(last_json))
        self.close()

class UpdateLocationsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Add Geographic Info')
        self.setMinimumSize(500, 400)
        layout = QVBoxLayout(self)

        self.csv_path = None
        self.csv_btn = QPushButton('Load CSV (default ChronAm_newspapers_XY.csv)')
        self.csv_btn.clicked.connect(self.load_csv)
        layout.addWidget(self.csv_btn)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        self.fetch_btn = QPushButton('Fetch Missing Metadata')
        self.fetch_btn.clicked.connect(self.fetch_metadata)
        layout.addWidget(self.fetch_btn)

        self.merge_btn = QPushButton('Merge to GeoJSON')
        self.merge_btn.clicked.connect(self.perform_merge)
        layout.addWidget(self.merge_btn)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load CSV', '', 'CSV Files (*.csv)')
        if not path:
            path = os.path.join(self.parent().project_folder, 'data', 'ChronAm_newspapers_XY.csv')
        self.csv_path = path
        missing = []
        try:
            import pandas as pd
            df = pd.read_csv(path)
            # Find newspapers with missing coordinates
            missing = df[df[['Long', 'Lat']].isna().any(axis=1)]['Title'].tolist()
        except Exception:
            pass
        self.list_widget.clear()
        for name in missing:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_widget.addItem(item)

    def fetch_metadata(self):
        to_fetch = [
            self.list_widget.item(i).text()
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == Qt.Checked
        ]
        if not self.csv_path:
            self.csv_path = os.path.join(self.parent().project_folder, 'data', 'ChronAm_newspapers_XY.csv')
        updated_csv = fetch_missing_metadata(self.parent().project_folder, to_fetch, self.csv_path)
        self.csv_path = updated_csv
        QMessageBox.information(self, 'Fetched', f'Metadata fetched and CSV updated:\n{updated_csv}')

    def perform_merge(self):
        parent = self.parent()
        json_path, _ = QFileDialog.getOpenFileName(self, 'Select JSON File', parent.project_folder, 'JSON Files (*.json)')
        if not json_path:
            return
        # Try to read minimal metadata from JSON (term and year)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            term = None
            year = None
            if isinstance(info, dict):
                term = info.get('search_term')
                year = info.get('year')
            if not term:
                fname = os.path.basename(json_path)
                base, _ = os.path.splitext(fname)
                term = base.split('_', 1)[0] if base else None
            if not term:
                raise ValueError('Could not infer search_term from JSON or filename.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Could not read metadata: {e}')
            return

        try:
            out_paths = merge_geojson(
                parent.project_folder,
                csv_path=self.csv_path,
                search_term=term,
                year=year,
                json_path=json_path
            )
            QMessageBox.information(self, 'GeoJSON Created', 'Files created:\n' + '\n'.join(out_paths))
            # Load the latest GeoJSON in the main window
            last_geo = out_paths[-1] if out_paths else None
            if last_geo:
                parent.geojson_file = last_geo
                parent.geojson_label.setText(os.path.basename(last_geo))
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

class CSVPreviewDialog(QDialog):
    def __init__(self, csv_path, parent=None, max_rows=100):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(csv_path))
        df = pd.read_csv(csv_path).head(max_rows)
        tbl = QTableWidget(df.shape[0], df.shape[1], self)
        tbl.setHorizontalHeaderLabels(list(df.columns))
        for i, row in df.iterrows():
            for j, val in enumerate(row):
                tbl.setItem(i, j, QTableWidgetItem(str(val)))
        layout = QVBoxLayout(self)
        layout.addWidget(tbl)

class CollocationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Collocation Analysis')
        self.setMinimumSize(620, 580)
        layout = QVBoxLayout(self)

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

        # Pre-fill term and dates from loaded JSON (if available)
        if parent.json_file:
            try:
                with open(parent.json_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                self.term_input.setText(info.get('search_term', ''))
                self.start_input.setText(info.get('start_date', ''))
                self.end_input.setText(info.get('end_date', ''))
            except Exception:
                pass

        # Time bin controls
        self.bin_size = QLineEdit('1')
        self.bin_size.setValidator(QIntValidator(1, 1000, self))
        self.bin_size.setMaximumWidth(60)
        form.addRow('Bin Size:', self.bin_size)
        self.bin_unit = QComboBox()
        self.bin_unit.addItems(['Days', 'Weeks', 'Months', 'Years'])
        form.addRow('Time Unit:', self.bin_unit)
        self.ignore_bin = QCheckBox('Ignore Bin Size (no time binning)')
        form.addRow(self.ignore_bin)

        # Additional collocation options (checkboxes)
        self.checks = {}
        for opt in ['include_page_count', 'include_first_last_date', 'include_cooccurrence_rate', 'include_relative_position', 'drop_stopwords']:
            cb = QCheckBox(opt)
            form.addRow(cb)
            self.checks[opt] = cb

        layout.addLayout(form)

        # Action buttons
        btn_run = QPushButton('Run Collocation')
        btn_bar = QPushButton('Show Bar Chart')
        btn_rank = QPushButton('Show Rank Changes')
        btn_run.clicked.connect(self.run_collocate)
        btn_bar.clicked.connect(self.show_bar)
        btn_rank.clicked.connect(self.show_rank)
        for b in (btn_run, btn_bar, btn_rank):
            layout.addWidget(b)

        # Initialize city/state dropdowns based on current mode
        self.on_mode_toggle()

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
                parent.geojson_label.setText(os.path.basename(p))
                # Update city/state lists for new GeoJSON
                self.populate_city_state()
        else:
            p, _ = QFileDialog.getOpenFileName(self, 'Select JSON Results', parent.project_folder, 'JSON Files (*.json)')
            if p:
                parent.json_file = p
                parent.json_label.setText(os.path.basename(p))
        # Update source label text
        self.source_label.setText(self._source_text())

    def on_mode_toggle(self):
        if self.mode_geo.isChecked():
            # Enable city/state filters
            self.city_combo.setEnabled(True)
            self.state_combo.setEnabled(True)
            # Populate dropdowns if a GeoJSON is loaded
            if getattr(self.parent(), 'geojson_file', None) and os.path.exists(self.parent().geojson_file):
                self.populate_city_state()
        else:
            # Disable filters for JSON mode
            self.city_combo.setEnabled(False)
            self.state_combo.setEnabled(False)
            # Reset selections to "All"
            self.city_combo.setCurrentIndex(0)
            self.state_combo.setCurrentIndex(0)

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

    def run_collocate(self):
        # Gather input parameters
        city_text = self.city_combo.currentText()
        state_text = self.state_combo.currentText()
        city = None if not city_text or city_text == 'All Cities' else city_text.strip()
        state = None if not state_text or state_text == 'All States' else state_text.strip()
        start = self.start_input.text().strip()
        end   = self.end_input.text().strip()
        term  = self.term_input.text().strip()

        # Determine time bin setting
        if self.ignore_bin.isChecked():
            time_bin = None
        else:
            size_text = self.bin_size.text().strip()
            if not size_text.isdigit():
                QMessageBox.warning(self, 'Invalid Bin Size', 'Please enter an integer ≥ 1.')
                return
            unit = self.bin_unit.currentText().lower()
            time_bin = f"{int(size_text)} {unit}"

        # Prepare output file labels (for user info)
        start_lbl = start or 'all'
        end_lbl   = end or 'all'
        safe_term = re.sub(r"[^A-Za-z0-9._-]", "", re.sub(r"\s+", "_", term)) or "term"
        metrics_csv = os.path.join(self.parent().project_folder, 'data', 'processed', f'collocates_metrics_{safe_term}_{start_lbl}_{end_lbl}.csv')
        by_time_csv = os.path.join(self.parent().project_folder, 'data', 'processed', f'collocates_by_time_{safe_term}_{start_lbl}_{end_lbl}.csv')

        # Run collocation analysis
        try:
            # Collect options from checkboxes
            opts = {opt: cb.isChecked() for opt, cb in self.checks.items()}
            if self.mode_json.isChecked():
                # JSON-only mode
                json_path = getattr(self.parent(), 'json_file', None)
                if not json_path or not os.path.exists(json_path):
                    self.choose_source_file()
                    json_path = getattr(self.parent(), 'json_file', None)
                    if not json_path:
                        return
                _ = run_collocation(
                    self.parent().project_folder,
                    city=city, state=state,
                    start_date=start or None, end_date=end or None,
                    term=term,
                    time_bin_unit=time_bin,
                    json_path=json_path,
                    geojson_path=None,
                    write_occurrences_geojson=False,
                    **opts
                )
                QMessageBox.information(self, 'Done', f'Collocation metrics saved:\n{metrics_csv}')
                preview = CSVPreviewDialog(metrics_csv, parent=self)
                preview.show()
            else:
                # GeoJSON mode
                geo_path = getattr(self.parent(), 'geojson_file', None)
                if not geo_path or not os.path.exists(geo_path):
                    self.choose_source_file()
                    geo_path = getattr(self.parent(), 'geojson_file', None)
                    if not geo_path:
                        return
                out_geo = run_collocation(
                    self.parent().project_folder,
                    city=city, state=state,
                    start_date=start or None, end_date=end or None,
                    term=term,
                    time_bin_unit=time_bin,
                    geojson_path=geo_path,
                    json_path=None,
                    write_occurrences_geojson=True,
                    **opts
                )
                QMessageBox.information(self, 'Done', f'Collocated occurrences GeoJSON:\n{out_geo}\n\nMetrics:\n{metrics_csv}')
                # Preview metrics CSV
                preview = CSVPreviewDialog(metrics_csv, parent=self)
                preview.show()
                # Update main window with new GeoJSON (collocated occurrences)
                if out_geo:
                    parent = self.parent()
                    parent.geojson_file = out_geo
                    parent.geojson_label.setText(os.path.basename(out_geo))
                # Optionally reveal the output file in OS file explorer
                try:
                    if sys.platform == "darwin":
                        subprocess.run(['open', '-R', out_geo])
                    elif sys.platform.startswith("win"):
                        os.startfile(out_geo)  # type: ignore[attr-defined]
                    else:
                        subprocess.run(['xdg-open', os.path.dirname(out_geo)])
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def show_bar(self):
        term  = self.term_input.text().strip()
        start = self.start_input.text().strip() or 'all'
        end   = self.end_input.text().strip() or 'all'
        safe_term = re.sub(r"[^A-Za-z0-9._-]", "", re.sub(r"\s+", "_", term)) or "term"
        csv_path = os.path.join(self.parent().project_folder, 'data', 'processed', f'collocates_metrics_{safe_term}_{start}_{end}.csv')
        plot_bar(csv_path)

    def show_rank(self):
        term = self.term_input.text().strip()
        start = self.start_input.text().strip() or 'all'
        end = self.end_input.text().strip() or 'all'
        safe_term = re.sub(r"[^A-Za-z0-9._-]", "", re.sub(r"\s+", "_", term)) or "term"
        file_path = os.path.join(self.parent().project_folder, 'data', 'processed', f'collocates_by_time_{safe_term}_{start}_{end}.csv')
        if not os.path.exists(file_path):
            QMessageBox.warning(self, 'No Data', 'Collocation by-time data not found. Please run collocation with a time bin first.')
            return
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Could not read by-time data: {e}')
            return
        if df.empty or 'time_bin' not in df.columns or 'collocate_term' not in df.columns or 'ordinal_rank' not in df.columns:
            QMessageBox.information(self, 'No Rank Data', 'No collocate rank data available for the selected parameters.')
            return
        # Determine time bins and prompt user for home bin and top-N
        try:
            unique_bins = sorted(df['time_bin'].unique(), key=lambda x: pd.to_datetime(str(x), errors='coerce'))
        except Exception:
            unique_bins = sorted(df['time_bin'].unique())
        num_bins = len(unique_bins)
        if num_bins == 0:
            QMessageBox.information(self, 'No Rank Data', 'No collocate rank data available.')
            return
        bin_num, ok = QInputDialog.getInt(self, 'Select Home Bin', f'Select a home rank bin (1 - {num_bins}):', 1, 1, num_bins)
        if not ok:
            return
        home_idx = bin_num - 1
        if home_idx < 0 or home_idx >= num_bins:
            return
        home_bin_label = unique_bins[home_idx]
        df_home = df[df['time_bin'] == home_bin_label]
        if df_home.empty:
            QMessageBox.information(self, 'No Data', 'The selected bin contains no collocates.')
            return
        max_terms = len(df_home)
        default_n = 10 if max_terms >= 10 else max_terms
        N, ok = QInputDialog.getInt(self, 'Select Top N', f'Enter top N terms to display (max {max_terms}):', default_n, 1, max_terms)
        if not ok:
            return
        N = int(N)
        # Get top-N collocate terms in the selected home bin
        df_home_sorted = df_home.sort_values('ordinal_rank')
        top_terms = df_home_sorted.head(N)['collocate_term'].tolist()
        df_top = df[df['collocate_term'].isin(top_terms)].copy()
        if df_top.empty:
            QMessageBox.information(self, 'No Data', 'No data available for the selected terms.')
            return
        plot_rank_changes(df_top)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())