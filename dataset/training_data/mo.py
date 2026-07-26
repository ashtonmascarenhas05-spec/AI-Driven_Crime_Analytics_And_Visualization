"""
feature_engineering.py
============================================================
MO Detection — Feature Engineering (Refined)
============================================================

Fixes applied vs. the earlier version:
  1. `primary_crime_head_encoded` and `victim_gender_mode` are now
     ONE-HOT encoded instead of being fed to StandardScaler as raw
     integer codes. DBSCAN is distance-based, so leaving them as plain
     numbers made it treat "category 2 vs category 11" as a real
     numeric gap — which is meaningless for nominal categories — and
     caused clusters to split almost entirely along category lines
     instead of actual behavioral similarity.
  2. Adds an explicit data-quality check that detects when `num_cases`
     and `geographic_spread_km` are constant (a symptom of the input
     being case-level rather than truly aggregated per accused person)
     and REFUSES to silently pretend those features are informative.
     It logs exactly what upstream data would be needed to fix this
     properly, and if a raw multi-case-per-person table is supplied,
     it performs the aggregation instead of relying on a pre-aggregated
     file that turns out not to actually be aggregated.

Input : mo_features.csv (or a raw case-level file — see AGGREGATION note)
Output: mo_features_engineered.csv
"""

import ast
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("feature_engineering")

# Columns that are nominal categories, not ordinal numbers — MUST be
# one-hot encoded rather than scaled as-is.
CATEGORICAL_COLS = ["primary_crime_head_encoded", "victim_gender_mode"]

# How many PCA components to compress the 75-dim Act/Section multi-hot
# vector down to. Bump this up if EXPLAINED_VARIANCE_TARGET isn't hit.
ACT_SECTION_PCA_COMPONENTS = 20
EXPLAINED_VARIANCE_TARGET = 0.40


# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
def load_data(input_path: str) -> pd.DataFrame:
    """Load the raw/semi-processed MO dataset from CSV."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Loading data from %s", input_path)
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Input file is empty.")
    if "person_uid" not in df.columns:
        raise ValueError("Expected column 'person_uid' not found in input file.")

    logger.info("Loaded %d rows, %d columns.", df.shape[0], df.shape[1])
    return df


# ------------------------------------------------------------------
# 2. Aggregate multi-case rows per accused person (only runs if needed)
# ------------------------------------------------------------------
def aggregate_to_person_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    If `person_uid` repeats (i.e. the file is at case-level, one row
    per case, not one row per accused person), roll it up so
    `num_cases` and `geographic_spread_km` become real, varying
    signals instead of constants.

    If the file already has one row per person_uid, this is a no-op
    EXCEPT it will loudly warn if num_cases/geographic_spread_km look
    constant, since that means the true aggregation happened even
    earlier upstream (or never happened) and this script can't recover
    the lost case-level detail from a file that no longer has it.
    """
    is_case_level = df["person_uid"].duplicated().any()

    if not is_case_level:
        logger.info(
            "person_uid is already unique per row — no row-level "
            "aggregation to perform here."
        )
        _warn_if_constant(df, "num_cases")
        _warn_if_constant(df, "geographic_spread_km")
        return df

    logger.info(
        "Detected repeated person_uid values — aggregating %d case rows "
        "up to person level.",
        len(df),
    )

    agg_spec = {}
    if "num_cases" in df.columns:
        # Real case count per person, not whatever placeholder value
        # existed at case level.
        agg_spec["num_cases"] = ("person_uid", "size")

    # Numeric behavioral columns: average across a person's cases.
    numeric_avg_cols = [
        c
        for c in ["pct_weekend", "avg_victim_age", "arrest_vs_surrender_ratio"]
        if c in df.columns
    ]

    grouped = df.groupby("person_uid")
    result = grouped.size().rename("num_cases").reset_index()

    for col in numeric_avg_cols:
        result = result.merge(
            grouped[col].mean().reset_index(), on="person_uid", how="left"
        )

    # Geographic spread requires per-case location data (lat/lon or
    # district code per case). If present, compute a real spread;
    # otherwise this can't be derived from case count alone.
    if {"case_lat", "case_lon"}.issubset(df.columns):
        result = result.merge(
            grouped.apply(_haversine_spread).rename("geographic_spread_km").reset_index(),
            on="person_uid",
            how="left",
        )
    elif "district" in df.columns:
        result = result.merge(
            grouped["district"].nunique().rename("district_count").reset_index(),
            on="person_uid",
            how="left",
        )
        logger.warning(
            "No lat/lon columns found — using distinct district count as a "
            "proxy for geographic spread instead of true distance in km."
        )
    else:
        logger.warning(
            "No location columns (case_lat/case_lon or district) found. "
            "geographic_spread_km cannot be computed and will be dropped. "
            "To fix this properly, the raw case table needs a location "
            "field per case."
        )

    # Categorical/mode columns: most frequent value across a person's cases.
    for col in ["primary_crime_head_encoded", "victim_gender_mode"]:
        if col in df.columns:
            mode_series = grouped[col].agg(lambda s: s.mode().iloc[0]).reset_index()
            result = result.merge(mode_series, on="person_uid", how="left")

    # act_section_encoded: union of all sections violated across a
    # person's cases (max, so any section hit in any case is flagged).
    if "act_section_encoded" in df.columns:
        def _union_sections(series):
            arrays = [np.array(ast.literal_eval(s), dtype=int) for s in series]
            return str(np.maximum.reduce(arrays).tolist())

        union_series = grouped["act_section_encoded"].agg(_union_sections).reset_index()
        result = result.merge(union_series, on="person_uid", how="left")

    logger.info("Aggregated to %d unique persons.", len(result))
    return result


def _haversine_spread(group: pd.DataFrame) -> float:
    """Max pairwise distance (km) between a person's case locations."""
    lats = np.radians(group["case_lat"].values)
    lons = np.radians(group["case_lon"].values)
    if len(lats) < 2:
        return 0.0
    max_dist = 0.0
    for i in range(len(lats)):
        for j in range(i + 1, len(lats)):
            dlat = lats[j] - lats[i]
            dlon = lons[j] - lons[i]
            a = np.sin(dlat / 2) ** 2 + np.cos(lats[i]) * np.cos(lats[j]) * np.sin(dlon / 2) ** 2
            max_dist = max(max_dist, 2 * 6371 * np.arcsin(np.sqrt(a)))
    return max_dist


def _warn_if_constant(df: pd.DataFrame, col: str) -> None:
    if col in df.columns and df[col].nunique() <= 1:
        logger.warning(
            "'%s' is constant across all %d rows — it will contribute "
            "ZERO signal to clustering. This usually means the source "
            "data is case-level rather than aggregated per real accused "
            "person. Fixing it requires the raw multi-case table (with a "
            "location field per case) rolled up by person_uid.",
            col,
            len(df),
        )


# ------------------------------------------------------------------
# 3. Parse & compress Act/Section multi-hot vector
# ------------------------------------------------------------------
def encode_act_sections(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the stringified multi-hot list and compress via PCA."""
    if "act_section_encoded" not in df.columns:
        logger.warning("'act_section_encoded' column not found — skipping.")
        return pd.DataFrame(index=df.index)

    logger.info("Parsing act_section_encoded...")
    act_matrix = np.vstack(
        df["act_section_encoded"].apply(lambda s: np.array(ast.literal_eval(s), dtype=int)).values
    )

    # Standardize the binary matrix before PCA so each of the 75
    # sections contributes proportionally rather than letting
    # high-frequency sections dominate the principal components.
    act_scaled = StandardScaler().fit_transform(act_matrix)

    n_components = min(ACT_SECTION_PCA_COMPONENTS, act_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    act_reduced = pca.fit_transform(act_scaled)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        "Act/Section PCA: %d components explain %.1f%% of variance.",
        n_components,
        explained * 100,
    )
    if explained < EXPLAINED_VARIANCE_TARGET:
        logger.warning(
            "Explained variance (%.1f%%) is below target (%.0f%%) — "
            "consider increasing ACT_SECTION_PCA_COMPONENTS.",
            explained * 100,
            EXPLAINED_VARIANCE_TARGET * 100,
        )

    return pd.DataFrame(
        act_reduced,
        columns=[f"act_pc_{i + 1}" for i in range(n_components)],
        index=df.index,
    )


# ------------------------------------------------------------------
# 4. One-hot encode nominal categoricals (THE core fix)
# ------------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode nominal category codes instead of leaving them as
    raw integers. This is the fix for clusters forming around category
    codes rather than real behavioral similarity.
    """
    present_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    if not present_cols:
        logger.warning("No categorical columns found to encode.")
        return pd.DataFrame(index=df.index)

    logger.info("One-hot encoding categorical columns: %s", present_cols)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[present_cols])
    return pd.DataFrame(
        encoded, columns=ohe.get_feature_names_out(present_cols), index=df.index
    )


# ------------------------------------------------------------------
# 5. Engineer + scale continuous numeric features
# ------------------------------------------------------------------
def engineer_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, derive, and scale the continuous numeric feature block."""
    work = df.copy()

    # avg_victim_age == 0 treated as a missing-value placeholder, not a
    # real age. Adjust this rule if 0 is a genuine age in your data.
    if "avg_victim_age" in work.columns:
        work["victim_age_missing"] = (work["avg_victim_age"] == 0).astype(int)
        median_age = work.loc[work["avg_victim_age"] > 0, "avg_victim_age"].median()
        work["avg_victim_age_clean"] = work["avg_victim_age"].replace(0, median_age)

    # Signed weekend preference: -1 (weekday-only) ... +1 (weekend-only)
    if "pct_weekend" in work.columns:
        work["weekend_pref_signed"] = (work["pct_weekend"] - 0.5) * 2

    # Case frequency is typically right-skewed once truly aggregated —
    # log-scale to stop a few high-frequency offenders from dominating
    # the distance metric.
    if "num_cases" in work.columns:
        work["num_cases_log"] = np.log1p(work["num_cases"])

    numeric_cols = [
        c
        for c in [
            "num_cases_log",
            "weekend_pref_signed",
            "avg_victim_age_clean",
            "victim_age_missing",
            "arrest_vs_surrender_ratio",
            "geographic_spread_km",
            "district_count",
        ]
        if c in work.columns
    ]

    if not numeric_cols:
        raise ValueError("No numeric feature columns available after engineering.")

    logger.info("Scaling numeric columns: %s", numeric_cols)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(work[numeric_cols])
    return pd.DataFrame(
        scaled, columns=[f"{c}_scaled" for c in numeric_cols], index=df.index
    )


# ------------------------------------------------------------------
# 6. Assemble + save
# ------------------------------------------------------------------
def assemble_features(
    person_uid: pd.Series,
    numeric_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
    act_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all engineered feature blocks into the final matrix."""
    final = pd.concat([numeric_df, categorical_df, act_df], axis=1)
    final.insert(0, "person_uid", person_uid.values)
    return final


def save_results(final_df: pd.DataFrame, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False)
    logger.info("Saved %d rows, %d columns to %s.", *final_df.shape, output_path)


# ------------------------------------------------------------------
# Pipeline entry point
# ------------------------------------------------------------------
def run_pipeline(
    input_path: str = "mo_features.csv",
    output_path: str = "mo_features_engineered.csv",
) -> dict:
    try:
        df = load_data(input_path)
        df = aggregate_to_person_level(df)

        person_uid = df["person_uid"]
        act_df = encode_act_sections(df)
        categorical_df = encode_categoricals(df)
        numeric_df = engineer_numeric_features(df)

        final_df = assemble_features(person_uid, numeric_df, categorical_df, act_df)
        save_results(final_df, output_path)

        return {
            "status": "success",
            "output_path": output_path,
            "shape": final_df.shape,
        }

    except FileNotFoundError as exc:
        logger.error("File error: %s", exc)
        return {"status": "error", "message": str(exc)}
    except ValueError as exc:
        logger.error("Data/validation error: %s", exc)
        return {"status": "error", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during feature engineering.")
        return {"status": "error", "message": f"Unexpected error: {exc}"}


def main(
    input_path: str = "mo_features.csv",
    output_path: str = "mo_features_engineered.csv",
) -> Optional[dict]:
    result = run_pipeline(input_path=input_path, output_path=output_path)
    if result["status"] == "error":
        logger.error("Pipeline failed: %s", result["message"])
        sys.exit(1)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MO Detection feature engineering.")
    parser.add_argument("--input", default="mo_features.csv")
    parser.add_argument("--output", default="mo_features_engineered.csv")
    args = parser.parse_args()

    main(input_path=args.input, output_path=args.output)