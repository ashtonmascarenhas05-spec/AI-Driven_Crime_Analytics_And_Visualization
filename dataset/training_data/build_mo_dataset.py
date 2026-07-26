import numpy as np
import pandas as pd

print("Loading CSV files...")
accused_df = pd.read_csv("./dataset/synthetic_data/Accused.csv")
case_df = pd.read_csv("./dataset/synthetic_data/CaseMaster.csv")
victim_df = pd.read_csv("./dataset/synthetic_data/Victim.csv")
arrest_df = pd.read_csv("./dataset/synthetic_data/ArrestSurrender.csv")
act_sec_df = pd.read_csv("./dataset/synthetic_data/ActSectionAssociation.csv")

print("Processing and merging features...")

# 1. Base mapping: Use AccusedMasterID as person_uid
df = accused_df.copy()
df.rename(columns={"AccusedMasterID": "person_uid"}, inplace=True)

# 2. Merge CaseMaster details for major head, dates, and coordinates
df = df.merge(
    case_df[
        [
            "CaseMasterID",
            "CrimeMajorHeadID",
            "IncidentFromDate",
            "latitude",
            "longitude",
        ]
    ],
    on="CaseMasterID",
    how="left",
)

# 3. Process Temporal Features
df["IncidentFromDate"] = pd.to_datetime(df["IncidentFromDate"], errors="coerce")
df["incident_hour"] = df["IncidentFromDate"].dt.hour
df["is_weekend"] = df["IncidentFromDate"].dt.dayofweek.isin([5, 6]).astype(int)

# Circular encoding for hour-of-day
df["hour_sin"] = np.sin(2 * np.pi * df["incident_hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["incident_hour"] / 24)

# 4. Merge Victim Demographics safely
def safe_mode(x):
    m = x.mode()
    return m.iloc[0] if not m.empty else 0

victim_agg = (
    victim_df.groupby("CaseMasterID")
    .agg(
        avg_victim_age=("AgeYear", "mean"),
        victim_gender_mode=("GenderID", safe_mode),
    )
    .reset_index()
)
df = df.merge(victim_agg, on="CaseMasterID", how="left")

# 5. Merge Arrest vs Surrender Records
arrest_agg = (
    arrest_df.groupby("AccusedMasterID")
    .agg(
        total_actions=("ArrestSurrenderID", "count"),
        arrest_count=("ArrestSurrenderTypeID", lambda x: (x == 1).sum()),
    )
    .reset_index()
)
arrest_agg.rename(columns={"AccusedMasterID": "person_uid"}, inplace=True)
df = df.merge(arrest_agg, on="person_uid", how="left")
df["arrest_vs_surrender_ratio"] = df["arrest_count"] / df["total_actions"].replace(0, 1)

# 6. Process Act + Section Associations (Multi-hot encoding representation)
act_sec_df["act_section_str"] = (
    "Act_" + act_sec_df["ActID"].astype(str) + "_Sec_" + act_sec_df["SectionID"].astype(str)
)
case_acts = pd.get_dummies(act_sec_df["act_section_str"]).groupby(act_sec_df["CaseMasterID"]).max()
df = df.merge(case_acts, on="CaseMasterID", how="left")

act_col_names = list(case_acts.columns)

# 7. Aggregate per person_uid safely
agg_dict = {
    "CaseMasterID": "count",
    "CrimeMajorHeadID": safe_mode,
    "hour_sin": "mean",
    "hour_cos": "mean",
    "is_weekend": "mean",
    "avg_victim_age": "mean",
    "victim_gender_mode": safe_mode,
    "latitude": "std",
    "longitude": "std",
    "arrest_vs_surrender_ratio": "mean",
}

for col in act_col_names:
    agg_dict[col] = "max"

mo_features = df.groupby("person_uid").agg(agg_dict).reset_index()

# Rename columns to match target model schema
mo_features.rename(
    columns={
        "CaseMasterID": "num_cases",
        "CrimeMajorHeadID": "primary_crime_head_encoded",
        "is_weekend": "pct_weekend",
    },
    inplace=True,
)

# Calculate geographic spread in km
mo_features["geographic_spread_km"] = np.sqrt(
    mo_features["latitude"].fillna(0) ** 2
    + mo_features["longitude"].fillna(0) ** 2
) * 111.0

mo_features["act_section_encoded"] = mo_features[act_col_names].values.tolist()
mo_features.drop(columns=act_col_names, inplace=True)
mo_features.drop(columns=["latitude", "longitude", "hour_sin", "hour_cos"], inplace=True)
mo_features.fillna(0, inplace=True)

# Export final model-ready dataset
output_path = "./dataset/mo_features.csv"
mo_features.to_csv(output_path, index=False)
print(f"Successfully generated complete master dataset at: {output_path}")
print(mo_features.head())