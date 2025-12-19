#!/bin/bash
set -euo pipefail

# Create a DMG that shows both ChronAmTool.app and an Applications symlink
# so users can drag-and-drop the app into /Applications.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"
APP_NAME="ChronAmTool.app"
VOL_NAME="ChronAmTool"
DMG_PATH="${DIST_DIR}/${VOL_NAME}.dmg"
STAGE_DIR="${DIST_DIR}/dmg_stage"

APP_PATH="${DIST_DIR}/${APP_NAME}"
if [ ! -d "${APP_PATH}" ]; then
  echo "App bundle not found at ${APP_PATH}. Build it with PyInstaller first." >&2
  exit 1
fi

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

cp -R "${APP_PATH}" "${STAGE_DIR}/"
ln -s /Applications "${STAGE_DIR}/Applications"

rm -f "${DMG_PATH}"
hdiutil create -quiet -volname "${VOL_NAME}" -srcfolder "${STAGE_DIR}" "${DMG_PATH}"

rm -rf "${STAGE_DIR}"
echo "Created DMG at ${DMG_PATH}"
