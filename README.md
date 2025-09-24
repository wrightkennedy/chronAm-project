# chronAm
Chronicling America Text Mining and Visualization Tool

This tool uses a local parquet dataset of the of the Chronicling America historical newspapers for keyword and date searches to build a corpus, collocation analysis, and geocoding (based on the city of the publisher). 

## Sources
The newspaper files are from AmericanStories (https://huggingface.co/datasets/dell-research-harvard/AmericanStories) Update (3/25/2025). The json files have been converted to parquet for efficient storage and searching. The local parquet files are different from the American Stories parquet files available on huggingface (https://huggingface.co/datasets/davanstrien/AmericanStories-parquet), which were based on version 1 of the AmericanStories dataset.

# First time set up
## Download Parquet Files
1. Download the newspaper articles stored as parquet datasets here: https://emailsc-my.sharepoint.com/:f:/r/personal/w_kennedy_sc_edu/Documents/data_tx?csf=1&web=1&e=gHy9xJ
2. If you downloaded the zip archive with all years between 1900 and 1922 ("AmericanStories_parquet1900-1922.zip"), unzip the archive.
3. In Finder/File Explorer, you should see parquet files for each year, roughly 2-3 GB each (for 1900-1910s)

## Set up the App (macOS)
VS Code Instructions for first-time use
VS Code needs Git installed and logged in (github account), if not done so already
1. Set folder location for program files using File > Open Folder... 
2. Clone Repository (https://github.com/wrightkennedy/chronAm-project.git)
    Cmd/Ctrl + Shift + P
    type `Clone` + Enter
    paste https://github.com/wrightkennedy/chronAm-project.git + Enter
3. Open Terminal with Ctrl + Shift + `
    The current directory should be set to the project folder by default
4. In the Terminal, create a new virtual environment with the following command `python3 -m venv venv`
5. Activate the virtual environment `source venv/bin/activate`
6. Install dependencies `pip install -r requirements.txt`
7. Run the python script `python app.py`
8. The first start up is often slow, since the software downloads the dependencies

## Set up the App (Windows)
VS Code Instructions for first-time use
1. Click Source Control tab on left
2. Click "Download Git for Windows"
    a. https://git-scm.com/downloads/win
    b. Download and install
    c. Choose the Default editor used by Git > "Use Visual Studio Code as Git's default editor"
    d. Use default selections for everything else in installer
3. In VS Code > Source Control (tab), click "Reload" to see Git
4. Select "Initialize Repository"
5. Cmd/Ctrl + Shift + P, then type "Clone"
    a. https://github.com/wrightkennedy/chronAm-project.git
    b. Select folder for clone
    c. (if first time using Git in VS Code, you will be asked to 'Sign in [to GitHub] with your browser"
    d. Would you like to open the repository or Add it to the Workspace > add it to the workspace.
6. Select the Explorer tab on the left
7. Top Menu > Terminal > New Terminal (or Ctrl + Shift + `)
    a. Select chronAm-project as the workspace; we will install the python virtual environment in this folder, and the .gitignore will make sure it is not uploaded to GitHub
    b. Windows (python -m venv venv)
        i. We noticed a new virtual environment; would you like to select it > "Yes"
8. To activate the venv on Windows, use `venv\Scripts\activate`
    a. If terminal returns a security error, do the following:
        i. Open PowerShell as Administrator
            1) Right-click the Start button and select Windows PowerShell (Admin) or search for "PowerShell," then right-click and choose "Run as Administrator."
        ii. Check current execution policy: Run the following command to check the current execution policy: `Get-ExecutionPolicy`
        iii. Change execution policy: To allow scripts to run, you need to set the execution policy to RemoteSigned (which allows locally created scripts to run). Run this command: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`
        iv. Return to VS Code and try activating the venv again: `.\venv\Scripts\Activate`
    b. To undo the change made to the execution policy:
        i. Open PowerShell as Administrator
            1) Right-click the Start button and select Windows PowerShell (Admin) or search for "PowerShell," then right-click and choose "Run as Administrator."
        ii. To set the execution policy back to its default value (Restricted), run the following command: `Set-ExecutionPolicy Restricted -Scope CurrentUser`
        iii. Verify the change to the execution policy: Run the following command to verify the current execution policy: `Get-ExecutionPolicy`

# Working in the app
## Create or Load a Project
1. Use File > New Project to create a new project folder (generally a good idea to keep it close to or in the chronam-project folder
2. In Finder/File Explorer, add a folder to chronam/data named parquet. `chronam/data/parquet/` and move the parquet files to this folder (downloaded in "Download Parquet Files").
3. [Optional] if you have already used the chronam software, use File > Open Project to open the folder location of the previous chronAm-project folder (this folder should contain your parquet folder with datasets).

## Search the Newspaper Datasets
1. Start by selecting "A) Search Dataset"
  1. Enter a Search Term into the box. Search is case-insensitive
  2. Enter a Start Date and End Date. Mind the format `[YYYY]-[MM]-[DD]`
     *Tip: start small - either use a less common term/phrase or a small date range*
2. Select "Run Download" *if using the parquet files, the tool runs locally and is not downloading new data*
3. Watch as the tool identifies and extracts all articles containing the search term within the date range.
4. The tool creates a JSON file in `data/raw/[search term]_[start date]-[end date].json`

## Add Geographic Info
1. 
