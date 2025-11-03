"""
topics.py
Topic modeling pipeline integrated with ChronAm filters.

Provides:
- run_topic_model: build topic models (LDA/NMF) from filtered article corpora.
- build_topic_model_output_paths: predict output locations mirroring collocation naming.

Requires scikit-learn when invoked.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import init_project  # type: ignore
from .utils import term_directory_name, write_metadata_file

# Reuse internal helpers from collocate to stay aligned with filtering/tokenization.
from .collocate import (  # type: ignore
    STOPWORDS,
    _assign_time_bin,
    _build_output_stem,
    _filter_df,
    _find_phrase_positions,
    _get_bin_edges,
    _load_geojson,
    _load_json,
    _normalize_term_groups,
    _suffix_with_groups,
    _tokenize,
)


@dataclass
class TopicModelParameters:
    model: str = "lda"
    n_topics: int = 10
    n_top_words: int = 12
    max_features: int = 3000
    min_df: float = 5
    max_df: float = 0.5
    max_documents: Optional[int] = None
    random_state: int = 0
    drop_stopwords: bool = False
    restrict_to_selected_terms: bool = False
    exclude_drop_term_documents: bool = False
    remove_drop_terms_from_tokens: bool = True
    min_topic_weight: float = 0.05
    max_topics_per_document: int = 3

    def as_payload(self) -> Dict[str, Any]:
        payload = self.__dict__.copy()
        # Keep floats limited for deterministic hashing while preserving precision.
        payload["min_topic_weight"] = float(self.min_topic_weight)
        payload["max_df"] = float(self.max_df)
        payload["min_df"] = float(self.min_df)
        return payload


def _ensure_sklearn(model: str):
    try:
        from sklearn.decomposition import LatentDirichletAllocation, NMF  # type: ignore
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # type: ignore
    except Exception as exc:  # pragma: no cover - surfaced via UI
        raise RuntimeError("Topic modeling requires scikit-learn. Install scikit-learn to continue.") from exc

    model = (model or "lda").strip().lower()
    if model not in {"lda", "nmf"}:
        raise ValueError("Unsupported topic model. Choose 'lda' or 'nmf'.")

    if model == "lda":
        return LatentDirichletAllocation, CountVectorizer
    return NMF, TfidfVectorizer


def _topic_suffix(params: TopicModelParameters, selected_terms: Sequence[str]) -> str:
    payload = {
        "params": params.as_payload(),
        "selected_terms": sorted({str(term).strip().lower() for term in selected_terms if str(term).strip()}),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"_topics_{digest}"


def _prepare_phrases(terms: Iterable[str]) -> List[List[str]]:
    phrases: List[List[str]] = []
    for term in terms or []:
        tokens = [tok for tok in _tokenize(str(term)) if tok]
        if tokens:
            phrases.append(tokens)
    return phrases


def _document_contains(tokens: List[str], phrases: List[List[str]]) -> bool:
    if not phrases:
        return False
    for phrase in phrases:
        if _find_phrase_positions(tokens, phrase):
            return True
    return False


def _remove_drop_tokens(tokens: List[str], drop_terms: Sequence[str]) -> List[str]:
    if not drop_terms:
        return tokens
    drop_single_tokens = {tok for term in drop_terms for tok in _tokenize(term)}
    if not drop_single_tokens:
        return tokens
    return [tok for tok in tokens if tok not in drop_single_tokens]


def _preprocess_documents(
    df: pd.DataFrame,
    *,
    params: TopicModelParameters,
    drop_terms: Sequence[str],
    selected_terms: Sequence[str],
) -> pd.DataFrame:
    drop_phrases = _prepare_phrases(drop_terms)
    selected_phrases = _prepare_phrases(selected_terms)

    rows = []
    for _, row in df.iterrows():
        text = row.get("article", "")
        if not isinstance(text, str) or not text.strip():
            continue
        tokens = _tokenize(text)
        if params.drop_stopwords:
            tokens = [tok for tok in tokens if tok not in STOPWORDS]

        drop_hit = _document_contains(tokens, drop_phrases)
        select_hit = _document_contains(tokens, selected_phrases)

        if params.exclude_drop_term_documents and drop_hit:
            continue
        if params.restrict_to_selected_terms and selected_phrases and not select_hit:
            continue

        if params.remove_drop_terms_from_tokens and drop_terms:
            tokens = _remove_drop_tokens(tokens, drop_terms)
        if not tokens:
            continue

        rows.append(
            {
                "article_id": row.get("article_id") or row.get("filename") or f"row{_}",
                "lccn": row.get("lccn") or row.get("SN"),
                "newspaper_name": row.get("newspaper_name"),
                "date": row.get("date"),
                "tokens": tokens,
                "text": " ".join(tokens),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["article_id", "text", "tokens", "date", "lccn", "newspaper_name"])
    return pd.DataFrame(rows)


def _limit_documents(df: pd.DataFrame, max_documents: Optional[int], random_state: int) -> pd.DataFrame:
    if not max_documents or max_documents <= 0 or len(df) <= max_documents:
        return df
    return df.sample(n=max_documents, random_state=random_state)


def build_topic_model_output_paths(
    project_dir: str,
    *,
    term: str,
    start_date: Optional[str],
    end_date: Optional[str],
    city: Optional[str],
    state: Optional[str],
    time_bin_unit: Optional[str],
    ignore_bin: bool,
    params: TopicModelParameters,
    drop_terms: Optional[Sequence[str]] = None,
    term_groups: Optional[List[dict]] = None,
    selected_terms: Optional[Sequence[str]] = None,
) -> Dict[str, Optional[str]]:
    paths = init_project(project_dir)
    processed_dir = paths["processed"]
    normalized_groups = _normalize_term_groups(term_groups)
    base_options = {
        "drop_stopwords": params.drop_stopwords,
        "ignore_drop_docs": params.exclude_drop_term_documents,
        "restrict_selected": params.restrict_to_selected_terms,
    }
    stem = _build_output_stem(
        term,
        start_date,
        end_date,
        city,
        state,
        time_bin_unit,
        ignore_bin,
        base_options,
    )
    topic_suffix = _topic_suffix(params, selected_terms or [])
    drop_suffix = _suffix_with_groups(drop_terms, normalized_groups)
    stem = f"{stem}{topic_suffix}{drop_suffix}"
    term_dir = os.path.join(processed_dir, term_directory_name(term))
    topics_path = os.path.join(term_dir, f"topics_{stem}.csv")
    doc_topics_path = os.path.join(term_dir, f"topic_documents_{stem}.csv")
    by_time_path = None if ignore_bin or not time_bin_unit else os.path.join(term_dir, f"topics_by_time_{stem}.csv")
    return {
        "stem": stem,
        "topics": topics_path,
        "doc_topics": doc_topics_path,
        "by_time": by_time_path,
    }


def _write_empty_outputs(
    output_paths: Dict[str, Optional[str]],
    project_dir: str,
    metadata_common: Dict[str, Any],
    metadata_enabled: bool,
) -> Dict[str, Optional[str]]:
    metadata_paths: Dict[str, str] = {}
    topics_path = output_paths.get("topics")
    doc_topics_path = output_paths.get("doc_topics")
    by_time_path = output_paths.get("by_time")

    if topics_path:
        os.makedirs(os.path.dirname(topics_path), exist_ok=True)
        pd.DataFrame(columns=["topic_id", "top_terms", "top_scores", "topic_weight"]).to_csv(topics_path, index=False)
        meta = dict(metadata_common)
        meta.update({"output_type": "topics_csv", "row_count": 0})
        meta_path = write_metadata_file(project_dir, topics_path, meta, enabled=metadata_enabled)
        if meta_path:
            metadata_paths["topics"] = meta_path
    if doc_topics_path:
        os.makedirs(os.path.dirname(doc_topics_path), exist_ok=True)
        pd.DataFrame(columns=["article_id", "topic_id", "weight"]).to_csv(doc_topics_path, index=False)
        meta = dict(metadata_common)
        meta.update({"output_type": "topic_documents_csv", "row_count": 0})
        meta_path = write_metadata_file(project_dir, doc_topics_path, meta, enabled=metadata_enabled)
        if meta_path:
            metadata_paths["doc_topics"] = meta_path
    if by_time_path:
        os.makedirs(os.path.dirname(by_time_path), exist_ok=True)
        pd.DataFrame(columns=["time_bin", "topic_id", "weight_sum", "doc_count", "ordinal_rank"]).to_csv(by_time_path, index=False)
        meta = dict(metadata_common)
        meta.update({"output_type": "topics_by_time_csv", "row_count": 0})
        meta_path = write_metadata_file(project_dir, by_time_path, meta, enabled=metadata_enabled)
        if meta_path:
            metadata_paths["by_time"] = meta_path

    return {
        "topics": topics_path,
        "doc_topics": doc_topics_path,
        "by_time": by_time_path,
        "metadata": metadata_paths,
    }


def _assign_time_bins(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str], unit_str: str) -> Tuple[pd.DataFrame, List[str]]:
    parts = unit_str.split()
    if len(parts) != 2 or not parts[0].isdigit():
        size, unit = 1, "months"
    else:
        size = max(1, int(parts[0]))
        unit = parts[1].lower()
    dates_series = pd.to_datetime(df["date"], errors="coerce")
    sdt = pd.to_datetime(start_date, errors="coerce") if start_date else dates_series.min()
    edt = pd.to_datetime(end_date, errors="coerce") if end_date else dates_series.max()
    if pd.isna(sdt) or pd.isna(edt):
        df["time_bin"] = None
        return df, []
    sdt = pd.to_datetime(sdt).normalize()
    edt = pd.to_datetime(edt).normalize()
    edges = _get_bin_edges(sdt, edt, unit, size)
    df = df.copy()
    df["time_bin"] = _assign_time_bin(df["date"], edges)
    labels = [edges[i].date().isoformat() for i in range(len(edges) - 1)]
    return df, labels


def _topic_top_terms(model, feature_names: Sequence[str], n_top_words: int) -> List[Dict[str, Any]]:
    components = getattr(model, "components_", None)
    if components is None:
        return []
    topics: List[Dict[str, Any]] = []
    for topic_idx, component in enumerate(components):
        # For NMF components may be sparse; convert to array.
        weights = np.array(component, dtype=float)
        top_indices = weights.argsort()[::-1][:n_top_words]
        top_terms = [feature_names[i] for i in top_indices]
        top_scores = [float(weights[i]) for i in top_indices]
        topics.append(
            {
                "topic_id": int(topic_idx),
                "top_terms": top_terms,
                "top_scores": top_scores,
                "topic_weight": float(weights.sum()),
            }
        )
    return topics


def _melt_doc_topics(
    matrix: np.ndarray,
    df_docs: pd.DataFrame,
    *,
    params: TopicModelParameters,
    topic_labels: Dict[int, str],
) -> pd.DataFrame:
    if matrix.size == 0:
        return pd.DataFrame(columns=["article_id", "topic_id", "weight"])

    indices = np.argsort(matrix, axis=1)[:, ::-1]
    weights_sorted = np.take_along_axis(matrix, indices, axis=1)
    topic_ids_sorted = indices

    records: List[Dict[str, Any]] = []
    for doc_idx in range(matrix.shape[0]):
        doc_row = df_docs.iloc[doc_idx]
        article_id = doc_row.get("article_id")
        date_val = doc_row.get("date")
        time_bin = doc_row.get("time_bin")
        lccn = doc_row.get("lccn")
        newspaper = doc_row.get("newspaper_name")

        for rank in range(min(params.max_topics_per_document, matrix.shape[1])):
            topic_id = int(topic_ids_sorted[doc_idx, rank])
            weight = float(weights_sorted[doc_idx, rank])
            if weight < params.min_topic_weight:
                if rank == 0:
                    # Always keep the dominant topic even if below threshold.
                    pass
                else:
                    continue
            records.append(
                {
                    "article_id": article_id,
                    "topic_id": topic_id,
                    "weight": weight,
                    "rank": rank + 1,
                    "date": date_val,
                    "time_bin": time_bin,
                    "lccn": lccn,
                    "newspaper_name": newspaper,
                    "topic_label": topic_labels.get(topic_id),
                }
            )
    if not records:
        return pd.DataFrame(columns=["article_id", "topic_id", "weight"])
    return pd.DataFrame(records)


def _build_topics_by_time(doc_topics: pd.DataFrame) -> pd.DataFrame:
    if doc_topics.empty or "time_bin" not in doc_topics.columns:
        return pd.DataFrame(columns=["time_bin", "topic_id", "weight_sum", "doc_count", "ordinal_rank"])

    working = doc_topics.dropna(subset=["time_bin"]).copy()
    if working.empty:
        return pd.DataFrame(columns=["time_bin", "topic_id", "weight_sum", "doc_count", "ordinal_rank"])

    grouped = (
        working.groupby(["time_bin", "topic_id"], as_index=False)
        .agg(weight_sum=("weight", "sum"), doc_count=("article_id", "nunique"))
    )
    grouped["ordinal_rank"] = (
        grouped.sort_values(["time_bin", "weight_sum", "topic_id"], ascending=[True, False, True])
        .groupby("time_bin")
        .cumcount()
        + 1
    )
    return grouped


def run_topic_model(
    project_dir: str,
    *,
    term: Optional[str],
    city: Optional[str] = None,
    state: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_bin_unit: Optional[str] = None,
    ignore_bin: bool = False,
    json_path: Optional[str] = None,
    geojson_path: Optional[str] = None,
    params: Optional[TopicModelParameters] = None,
    drop_terms: Optional[Sequence[str]] = None,
    term_groups: Optional[List[dict]] = None,
    selected_terms: Optional[Sequence[str]] = None,
    metadata_enabled: bool = True,
) -> Dict[str, Optional[str]]:
    if not json_path and not geojson_path:
        raise ValueError("Provide either json_path or geojson_path for topic modeling.")
    if not term:
        raise ValueError("Search term is required to run topic modeling.")

    params = params or TopicModelParameters()
    drop_terms = [str(term).strip() for term in drop_terms or [] if str(term).strip()]
    selected_terms = [str(term).strip() for term in selected_terms or [] if str(term).strip()]
    normalized_groups = _normalize_term_groups(term_groups)

    ModelClass, VectorizerClass = _ensure_sklearn(params.model)

    is_geo = bool(geojson_path)
    if is_geo:
        df_raw = _load_geojson(geojson_path)  # type: ignore[arg-type]
    else:
        df_raw = _load_json(json_path)  # type: ignore[arg-type]

    df_filtered = _filter_df(df_raw, start_date, end_date, city, state, is_geo=is_geo)

    metadata_common = {
        "tool": "topic_model",
        "parameters": {
            "city": city,
            "state": state,
            "start_date": start_date,
            "end_date": end_date,
            "term": term,
            "time_bin_unit": None if ignore_bin else time_bin_unit,
            "ignore_bin": bool(ignore_bin),
            "topic_params": params.as_payload(),
            "drop_terms": list(drop_terms),
            "selected_terms": list(selected_terms),
            "term_groups": [
                {"name": group["name"], "terms": list(group["terms"])} for group in normalized_groups
            ],
        },
        "inputs": {
            "json_path": json_path,
            "geojson_path": geojson_path,
        },
    }

    output_paths = build_topic_model_output_paths(
        project_dir,
        term=term,
        start_date=start_date,
        end_date=end_date,
        city=city,
        state=state,
        time_bin_unit=time_bin_unit,
        ignore_bin=ignore_bin,
        params=params,
        drop_terms=drop_terms,
        term_groups=normalized_groups,
        selected_terms=selected_terms,
    )

    if df_filtered.empty:
        return _write_empty_outputs(output_paths, project_dir, metadata_common, metadata_enabled)

    df_prepared = _preprocess_documents(
        df_filtered,
        params=params,
        drop_terms=drop_terms,
        selected_terms=selected_terms,
    )
    if df_prepared.empty:
        return _write_empty_outputs(output_paths, project_dir, metadata_common, metadata_enabled)

    df_prepared = _limit_documents(df_prepared, params.max_documents, params.random_state)

    if not ignore_bin and time_bin_unit:
        df_prepared, _ = _assign_time_bins(df_prepared, start_date, end_date, time_bin_unit)
    else:
        df_prepared = df_prepared.assign(time_bin=None)

    try:
        min_df_numeric = float(params.min_df)
    except (TypeError, ValueError):
        min_df_numeric = 1.0
    if min_df_numeric >= 1.0 and abs(min_df_numeric - round(min_df_numeric)) < 1e-9:
        min_df_value = int(round(min_df_numeric))
    else:
        min_df_value = max(0.0, min_df_numeric)

    try:
        max_df_numeric = float(params.max_df)
    except (TypeError, ValueError):
        max_df_numeric = 1.0
    if max_df_numeric <= 0.0:
        max_df_numeric = 0.05
    if max_df_numeric > 1.0:
        max_df_numeric = 1.0

    vectorizer = VectorizerClass(
        max_features=params.max_features,
        min_df=min_df_value,
        max_df=max_df_numeric,
        dtype=np.float64,
    )
    matrix = vectorizer.fit_transform(df_prepared["text"])
    feature_names = vectorizer.get_feature_names_out()

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return _write_empty_outputs(output_paths, project_dir, metadata_common, metadata_enabled)

    if params.model == "lda":
        model = ModelClass(
            n_components=params.n_topics,
            learning_method="online",
            max_iter=20,
            random_state=params.random_state,
        )
        transformed = model.fit_transform(matrix)
    else:
        model = ModelClass(
            n_components=params.n_topics,
            init="nndsvda",
            max_iter=400,
            random_state=params.random_state,
        )
        transformed = model.fit_transform(matrix)

    topics = _topic_top_terms(model, feature_names, params.n_top_words)
    topic_labels = {topic["topic_id"]: ", ".join(topic["top_terms"][:5]) for topic in topics}

    doc_topics = _melt_doc_topics(
        transformed,
        df_prepared.reset_index(drop=True),
        params=params,
        topic_labels=topic_labels,
    )

    if doc_topics.empty:
        return _write_empty_outputs(output_paths, project_dir, metadata_common, metadata_enabled)

    topics_df = pd.DataFrame(topics)
    topics_df["top_terms"] = topics_df["top_terms"].apply(lambda terms: ", ".join(terms))
    topics_df["top_scores"] = topics_df["top_scores"].apply(lambda scores: ", ".join(f"{score:.4f}" for score in scores))

    os.makedirs(os.path.dirname(output_paths["topics"] or ""), exist_ok=True)
    topics_df.to_csv(output_paths["topics"], index=False)

    doc_topics.to_csv(output_paths["doc_topics"], index=False)

    metadata_paths: Dict[str, str] = {}
    topics_meta = dict(metadata_common)
    topics_meta.update(
        {
            "output_type": "topics_csv",
            "row_count": int(len(topics_df)),
            "training_documents": int(len(df_prepared)),
            "vocabulary_size": int(len(feature_names)),
        }
    )
    meta_path = write_metadata_file(project_dir, output_paths["topics"], topics_meta, enabled=metadata_enabled)
    if meta_path:
        metadata_paths["topics"] = meta_path

    doc_topics_meta = dict(metadata_common)
    doc_topics_meta.update(
        {
            "output_type": "topic_documents_csv",
            "row_count": int(len(doc_topics)),
        }
    )
    meta_path = write_metadata_file(project_dir, output_paths["doc_topics"], doc_topics_meta, enabled=metadata_enabled)
    if meta_path:
        metadata_paths["doc_topics"] = meta_path

    by_time_path = output_paths.get("by_time")
    if by_time_path:
        by_time = _build_topics_by_time(doc_topics)
        by_time.to_csv(by_time_path, index=False)
        by_time_meta = dict(metadata_common)
        by_time_meta.update(
            {
                "output_type": "topics_by_time_csv",
                "row_count": int(len(by_time)),
            }
        )
        meta_path = write_metadata_file(project_dir, by_time_path, by_time_meta, enabled=metadata_enabled)
        if meta_path:
            metadata_paths["by_time"] = meta_path

    return {
        "topics": output_paths.get("topics"),
        "doc_topics": output_paths.get("doc_topics"),
        "by_time": output_paths.get("by_time"),
        "metadata": metadata_paths,
    }
