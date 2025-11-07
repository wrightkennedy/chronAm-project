# ChronAm Distribution Notes

Thanks for trying ChronAm! Depending on how you received the project, pick the option below to get started.

## Option A: macOS App Bundle (recommended)
1. Double-click `ChronAmTool.dmg` and drag **ChronAmTool.app** into `/Applications` (or any folder you prefer).
2. On first launch, macOS may flag the app as downloaded from the internet. Control-click the app, choose **Open**, then confirm.
3. During the first launch ChronAm creates `~/Documents/ChronAm/` (if it does not already exist) and copies the bundled sample dataset there. The app automatically targets `~/Documents/ChronAm/data/parquet/`, so you can immediately run a search against `AmericanStories_1800.parquet`.
4. Swap in your full dataset anytime via **Sources ▸ Set Local Dataset Folder…**.
   - For the full corpus, download the year-by-year parquet files from the Wright Kennedy SharePoint collection (access required): https://emailsc-my.sharepoint.com/:f:/r/personal/w_kennedy_sc_edu/Documents/data_tx?csf=1&web=1&e=nHPsNm
5. Switch between projects or data folders at any time via **File ▸ Preferences** in the app.

## Option B: Run from source (macOS, Linux, or Windows)
1. Unzip the repository anywhere you have write access.
2. Double-click `bootstrap.sh` (macOS/Linux) or `bootstrap.cmd` (Windows). The helper script will create an isolated `chronam-env` virtual environment, install `requirements.txt`, and launch the GUI.
   - If your system asks for permission to execute the script, allow it (macOS: `chmod +x bootstrap.sh` then run from Terminal).
3. Place your parquet dataset outside the repo (for example `~/ChronAmData/parquet/` on macOS or `D:\ChronAmData\parquet\` on Windows) to avoid bloating git.
4. When the UI starts it will still auto-configure `~/Documents/ChronAm/data/parquet` if the bundled sample is available; change it via **Sources ▸ Set Local Dataset Folder…** if you prefer another location.

### Dataset reminders
- Keep the large parquet files outside any synced folder you plan to share; the app only needs a path.
- Only `AmericanStories_1800.parquet` is bundled for demos. Use the SharePoint link above (or the Hugging Face mirror if available) for additional years.

Need help? Open an issue or ping the maintainer with details about your platform and what you tried. Happy digging!
