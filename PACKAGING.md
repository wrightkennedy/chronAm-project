# Packaging ChronAm Releases

This document records the exact steps used to create distributable builds.

## 1. Create a clean build environment
```bash
python3 -m venv build-env
source build-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt pyinstaller
```

## 2. Update assets (only if needed)
- Icons live in `chronam/resources/app_icon.png` and are converted into `app.icns`/`app.ico`.
- Reference CSVs in `chronam/resources/` and the sample `data/parquet/AmericanStories_1800.parquet` are embedded via `ChronAmTool.spec`.

## 3. Freeze the application
```bash
source build-env/bin/activate
pyinstaller ChronAmTool.spec
```
Outputs:
- `dist/ChronAmTool.app` — macOS bundle
- `dist/ChronAmTool` — raw onedir folder (handy for debugging)

To produce a Windows build, run the same command on a Windows machine (PyInstaller is not cross-compiling). The spec automatically selects `app.ico` and builds a console-less `ChronAmTool.exe`.

## 4. Wrap macOS bundle into a DMG
Use the helper script so the DMG contains both the app bundle and an `Applications` shortcut (standard drag-and-drop install experience).
```bash
./scripts/build_dmg.sh
```

## 5. Prepare the source distribution
- Ensure `bootstrap.sh` (macOS/Linux) and `bootstrap.cmd` (Windows) sit at the repo root.
- Include `DIST_README.md` alongside the repo zip to explain dataset setup and entry points.
- Confirm the bundled sample parquet (`data/parquet/AmericanStories_1800.parquet`) exists before building; the app copies it into `~/Documents/ChronAm/data/parquet/` on first launch.

## 6. Smoke test
- Launch `dist/ChronAmTool.app` and run a quick search using the bundled `AmericanStories_1800.parquet`.
- From source, run `./bootstrap.sh` (macOS/Linux) or `bootstrap.cmd` (Windows) and ensure the GUI starts.
- After first launch, verify that `~/Documents/ChronAm/` exists, contains the sample parquet, and is selected automatically inside the app.

## 7. Ship
- Share `dist/ChronAmTool.dmg` with macOS testers.
- Zip the repo (excluding `build-env`, `build`, `dist`) plus the bootstrap scripts and `DIST_README.md` for colleagues who prefer running from source or on Windows.

Version control reminder: never commit `build-env/`, `dist/`, `ChronAmTool.dmg`, or large parquet datasets. Only keep the scripts/specs that describe how to reproduce the artifacts.
