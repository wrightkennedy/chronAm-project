import os
import json
import re
from datetime import datetime, date
from datasets import load_dataset
import pandas as pd
from .config import init_project, DATASET_NAME, DATASET_CONFIG_NAME

def download_data(
    project_dir: str,
    search_term: str,
    start_date_str: str,
    end_date_str: str,
    heartbeat_interval: int = 10000,
    max_saved_articles_per_year: int = None,
    batch_size: int = 10000,
    progress_callback=None,
    cancel_event=None,
) -> list:
    paths = init_project(project_dir)
    output_dir = paths['raw']

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    if end_date < start_date:
        raise ValueError("end_date must be on/after start_date")

    years = [str(y) for y in range(start_date.year, end_date.year + 1)]
    ds_dict = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG_NAME,
        year_list=years,
        streaming=True,
        trust_remote_code=True
    )

    pattern = re.compile(re.escape(search_term), re.IGNORECASE)
    out_paths = []

    for year_str in years:
        window_start = max(start_date, date(int(year_str), 1, 1))
        window_end = min(end_date, date(int(year_str), 12, 31))
        if window_start > window_end:
            continue
        ws_str = window_start.isoformat()
        we_str = window_end.isoformat()

        ds = ds_dict[year_str]
        matches = []
        checked = 0
        batch = []
        limit_reached = False

        for ex in ds:
            checked += 1
            if heartbeat_interval and checked % heartbeat_interval == 0 and progress_callback:
                progress_callback(checked)

            batch.append(ex)
            if len(batch) < batch_size and not limit_reached:
                continue

            df_batch = pd.DataFrame(batch)
            if 'date' not in df_batch.columns or 'article' not in df_batch.columns:
                batch.clear()
                if limit_reached:
                    break
                continue

            date_series = df_batch['date'].fillna('')
            mask_date = date_series.between(ws_str, we_str)
            article_series = df_batch['article'].fillna('')
            mask_text = article_series.str.contains(pattern, na=False)
            mask = mask_date & mask_text

            for idx in mask[mask].index:
                matches.append(batch[idx])
                if max_saved_articles_per_year and len(matches) >= max_saved_articles_per_year:
                    limit_reached = True
                    break

            batch.clear()
            if limit_reached:
                break

        if not limit_reached and batch:
            df_batch = pd.DataFrame(batch)
            if 'date' in df_batch.columns and 'article' in df_batch.columns:
                date_series = df_batch['date'].fillna('')
                mask_date = date_series.between(ws_str, we_str)
                article_series = df_batch['article'].fillna('')
                mask_text = article_series.str.contains(pattern, na=False)
                mask = mask_date & mask_text

                for idx in mask[mask].index:
                    matches.append(batch[idx])
                    if max_saved_articles_per_year and len(matches) >= max_saved_articles_per_year:
                        break

        out_file = os.path.join(output_dir, f"{search_term}_{year_str}.json")
        payload = {
            "year": year_str,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "search_term": search_term,
            "match_count": len(matches),
            "articles": matches
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        out_paths.append(out_file)

    return out_paths
