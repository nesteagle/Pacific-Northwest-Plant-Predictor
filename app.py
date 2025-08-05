import os
from collections import defaultdict
from datetime import datetime

import gradio as gr
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from datasets import load_dataset

# Cache file for processed dataframe
CACHE_FILE = "processed_occurrences.pkl"

# Columns you want to load and process
usecols = [
    "DecimalLatitude",
    "DecimalLongitude",
    "ValidLatLng",
    "Family",
    "ScientificName",
    "YearCollected",
    "MinimumElevationInMeters",
    "MaximumElevationInMeters",
]

LAT_COL = "DecimalLatitude"
LON_COL = "DecimalLongitude"
MIN_LAT, MAX_LAT = 40.0, 70
MIN_LON, MAX_LON = -160.0, -110.0


def process_dataset(dataset, usecols):
    lines = [line for line in dataset["text"] if line.strip()]
    header = lines[0].split("\t")
    data_lines = lines[1:]
    rows = [line.split("\t") for line in data_lines]
    df_full = pd.DataFrame(rows, columns=header)
    cols_to_use = [col for col in usecols if col in df_full.columns]
    df = df_full[cols_to_use].copy()
    return df


def get_bounded_df():
    if os.path.exists(CACHE_FILE):
        df = pd.read_pickle(CACHE_FILE)
        print("Loaded cached processed data.")
    else:
        dataset = load_dataset(
            "nesteagle/CPNWH",
            data_files={"occurrences": "occurrences.txt"},
            split="occurrences",
        )

        # tsv to pandas
        df = process_dataset(dataset, column_names=usecols)

        df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
        df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
        df["YearCollected"] = pd.to_numeric(df["YearCollected"], errors="coerce")
        df["MinimumElevationInMeters"] = pd.to_numeric(
            df["MinimumElevationInMeters"], errors="coerce"
        )
        df["MaximumElevationInMeters"] = pd.to_numeric(
            df["MaximumElevationInMeters"], errors="coerce"
        )

        # Filter on lat/lon bounds and drop rows with missing lat/lon
        df = df[
            (df[LAT_COL] >= MIN_LAT)
            & (df[LAT_COL] <= MAX_LAT)
            & (df[LON_COL] >= MIN_LON)
            & (df[LON_COL] <= MAX_LON)
        ].dropna(subset=[LAT_COL, LON_COL])

        # Save to cache file
        df.to_pickle(CACHE_FILE)
        print("Processed and cached data saved.")
    return df


df_bounded = get_bounded_df()

# filter for non-null ScientificName cols
df_bounded = df_bounded[df_bounded["ScientificName"].notna()]

# filter out rare species with less than 100 occurrences total
species_counts = df_bounded["ScientificName"].value_counts()
species_to_keep = species_counts[species_counts >= 100].index

print("Total species:", df_bounded["ScientificName"].nunique())
print("Species to keep:", len(species_to_keep))

df_filtered = df_bounded[df_bounded["ScientificName"].isin(species_to_keep)]

# get records within 40 years prev
df_filtered = df_filtered[
    df_filtered["YearCollected"] >= datetime.now().year - 40
].copy()

coords = df_filtered[["DecimalLatitude", "DecimalLongitude"]].values
labels = df_filtered["ScientificName"].values
tree = KDTree(coords)

GRID_SIZE = 0.01  # in degrees

df_filtered = df_filtered.copy()
df_filtered.loc[:, "MeanElevation"] = (
    df_filtered["MinimumElevationInMeters"] + df_filtered["MaximumElevationInMeters"]
) / 2
df_filtered.loc[:, "grid_x"] = (df_filtered["DecimalLongitude"] // GRID_SIZE).astype(
    int
)
df_filtered.loc[:, "grid_y"] = (df_filtered["DecimalLatitude"] // GRID_SIZE).astype(int)

agg = (
    df_filtered.groupby(["grid_x", "grid_y"])
    .agg(
        species_richness=("ScientificName", "nunique"),
        dominant_species_count=("ScientificName", lambda x: x.value_counts().max()),
    )
    .reset_index()
)

df_with_grid = df_filtered.merge(agg, on=["grid_x", "grid_y"], how="left")


# distance of 0.1deg -> roughly 7x11km
def predict_plants(lat, lon, max_distance=0.05):
    indices = tree.query_ball_point([lat, lon], r=max_distance)

    if not indices:
        return None

    neighbors = labels[indices]

    distances = ((coords[indices] - np.array([lat, lon])) ** 2).sum(axis=1) ** 0.5
    weights = 1 / (distances + 1e-6)

    species_weights = defaultdict(float)
    for sp, w in zip(neighbors, weights):
        species_weights[sp] += w

    total_weight = sum(species_weights.values())
    results = [
        (species, 100 * weight / total_weight)
        for species, weight in species_weights.items()
    ]
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return "No plants nearby"
    str_results = [
        f"{species}, {(100 * weight / total_weight):.2f}%)\n"
        for species, weight in results
    ]
    return str_results


def predict(lat, lon):
    return predict_plants(lat=lat, lon=lon)


demo = gr.Interface(fn=predict, inputs="text", outputs="text")
demo.launch()
