"""Predicts likely plant species at given latitude and longitude coordinates and hosts a Gradio interface"""

import os
from collections import defaultdict
from datetime import datetime

import gradio as gr
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from datasets import load_dataset

CACHE_FILE = "processed_occurrences.pkl"

USE_COLS = [
    "DecimalLatitude",
    "DecimalLongitude",
    "ValidLatLng",
    "Family",
    "ScientificName",
    "YearCollected",
]

LAT_COL = "DecimalLatitude"
LON_COL = "DecimalLongitude"
MIN_LAT, MAX_LAT = 40.0, 70
MIN_LON, MAX_LON = -160.0, -110.0


def process_dataset(dataset, usecols):
    """Convert dataset text format to pandas DataFrame."""
    lines = [line for line in dataset["text"] if line.strip()]
    header = lines[0].split("\t")
    data_lines = lines[1:]
    rows = [line.split("\t") for line in data_lines]
    df_full = pd.DataFrame(rows, columns=header)
    cols_to_use = [col for col in usecols if col in df_full.columns]
    df = df_full[cols_to_use].copy()
    return df


def get_bounded_df():
    """Load and preprocess herbarium data."""
    if os.path.exists(CACHE_FILE):
        df = pd.read_pickle(CACHE_FILE)
        print("Loaded cached processed data.")
    else:
        dataset = load_dataset(
            "nesteagle/CPNWH",
            data_files={"occurrences": "occurrences.txt"},
            split="occurrences",
        )

        # process to pandas
        df = process_dataset(dataset, USE_COLS)

        df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
        df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
        df["ValidLatLng"] = df["ValidLatLng"].astype(str).str.lower().eq("true")

        # Filter on lat/lon bounds and drop rows with missing or invalid lat/lon
        df = df[
            (df["ValidLatLng"])
            & (df[LAT_COL] >= MIN_LAT)
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

# filter out rare species with less than 15 occurrences total
species_counts = df_bounded["ScientificName"].value_counts()
species_to_keep = species_counts[species_counts >= 15].index

df_filtered = df_bounded[df_bounded["ScientificName"].isin(species_to_keep)]

# get records within 40 years prev
df_filtered = df_filtered[
    df_filtered["YearCollected"] >= datetime.now().year - 40
].copy()

coords = df_filtered[["DecimalLatitude", "DecimalLongitude"]].values
labels = df_filtered["ScientificName"].values
tree = KDTree(coords)

GRID_SIZE = 0.01  # in degrees latitude/longitude

df_filtered = df_filtered.copy()
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
def predict_plants(lat, lon, max_distance=0.05, top_k=20):
    """Predict plant species likelihood at given coordinates using spatial K-NN."""
    indices = tree.query_ball_point([lat, lon], r=max_distance)
    if not indices:
        return None

    neighbors = labels[indices]
    neighbor_coords = coords[indices]

    # Compute Euclidean distance and inverse-distance weights
    deltas = neighbor_coords - np.array([lat, lon])
    distances = np.linalg.norm(deltas, axis=1)
    weights = 1 / (distances + 1e-6)

    # collect weight by species
    species_weights = defaultdict(float)
    for species, weight in zip(neighbors, weights):
        species_weights[species] += weight

    # sort top_k species by weight
    total_weight = sum(species_weights.values())
    if total_weight == 0:
        return None

    results = [
        (species_name, 100 * weight / total_weight)
        for species_name, weight in species_weights.items()
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def predict(lat, lon):
    """Main prediction interface"""
    predictions = predict_plants(lat=lat, lon=lon)
    return format_predictions(predictions=predictions)


def format_predictions(predictions):
    """Formats predictions for display"""
    if not predictions:
        return "No plants found nearby."
    lines = []
    for species, prob in predictions:
        name = species if species.strip() else "(Unknown species)"
        lines.append(f"{name}: {prob:.2f}%")
    return "\n".join(lines)


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label="Latitude"), gr.Number(label="Longitude")],
    outputs="text",
    title="Latitude/Longitude prediction",
)
demo.launch()
