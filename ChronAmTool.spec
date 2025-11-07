# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

spec_path = Path(globals().get("__file__", "ChronAmTool.spec"))
project_root = spec_path.resolve().parent if spec_path.exists() else Path.cwd()
resources_dir = project_root / "chronam" / "resources"
icon = resources_dir / ("app.icns" if sys.platform == "darwin" else "app.ico")
is_mac = sys.platform == "darwin"
hidden_imports = [
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_qtagg",
]

datas = [
    (str(resources_dir / "ChronAm_newspapers_XY.csv"), "chronam/resources"),
    (str(resources_dir / "ChronAm_yearly_dataset_summary.csv"), "chronam/resources"),
]

sample_parquet = project_root / "data" / "parquet" / "AmericanStories_1800.parquet"
if sample_parquet.exists():
    datas.append((str(sample_parquet), "data/parquet"))

a = Analysis(
    ["app.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ChronAmTool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=str(icon),
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ChronAmTool",
)

if is_mac:
    app = BUNDLE(
        coll,
        name="ChronAmTool.app",
        icon=str(icon),
        bundle_identifier="org.chronam.tool",
    )
