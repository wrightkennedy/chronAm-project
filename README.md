# chronAm
Chronicling America Text Mining and Visualization Tool

This tool uses a local parquet dataset of the of the Chronicling America historical newspapers for keyword and date searches to build a corpus, collocation analysis, and geocoding (based on the city of the publisher). 

## Sources
The newspaper files are from AmericanStories (https://huggingface.co/datasets/dell-research-harvard/AmericanStories) Update (3/25/2025). The json files have been converted to parquet for efficient storage and searching. The local parquet files are different from the American Stories parquet files available on huggingface (https://huggingface.co/datasets/davanstrien/AmericanStories-parquet), which were based on version 1 of the AmericanStories dataset.

# First time set up
## Download Parquet Files
1. Download the newspaper articles stored as parquet files here: https://emailsc-my.sharepoint.com/:u:/r/personal/w_kennedy_sc_edu/Documents/data_tx/AmericanStories_1910.parquet?csf=1&web=1&e=5deX4n
2. If you downloaded the zip archive with all years between 1900 and 1922, unzip the archive.
3. You should see parquet files for each year, roughly 2-3 GB each (for 1900-1910s)

## Set up the App
1. Clone Repository
2. Create a new virtual environment `python3 -m venv venv`
3. Activate the virtual environment `source venv/bin/activate`
4. Install dependencies `pip install -r requirements.txt`
5. Run the python script `python app.py`
6. The first start up is often slow, since the software downloads the NLTK tools and stopwords (even though stop words from NLTK are not working right now -- they are hardcoded)

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
