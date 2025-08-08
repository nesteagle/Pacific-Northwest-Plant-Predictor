"""Predicts likely plant species at given latitude and longitude coordinates and hosts a Gradio interface"""

from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass

import gradio as gr
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from datasets import load_dataset

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

MIN_OCCURRENCES = 15
YEARS_BACK = 40


def get_bounded_df():
    """Load and preprocess herbarium data; enforce dtypes and bounds."""
    dataset = load_dataset(
        "nesteagle/CPNWH",
        data_files={"occurrences": "occurrences.txt"},
        split="occurrences",
    )

    lines = [line for line in dataset["text"] if line.strip()]
    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:]]
    df_full = pd.DataFrame(rows, columns=header)
    df = df_full[[c for c in USE_COLS if c in df_full.columns]].copy()

    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    df["ValidLatLng"] = (
        df["ValidLatLng"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "t", "1", "yes", "y"])
    )
    df["YearCollected"] = pd.to_numeric(df["YearCollected"], errors="coerce")

    df = df[
        (df["ValidLatLng"])
        & (df[LAT_COL].between(MIN_LAT, MAX_LAT, inclusive="both"))
        & (df[LON_COL].between(MIN_LON, MAX_LON, inclusive="both"))
    ].dropna(subset=[LAT_COL, LON_COL])

    print(f"Loaded & bounded rows: {len(df)}")
    return df


@dataclass
class SpeciesIndex:
    coords: np.ndarray
    labels: np.ndarray
    tree: KDTree


def build_index() -> SpeciesIndex | None:
    """Load, filter, and build KDTree index. Returns None if no data."""
    df_bounded = get_bounded_df()

    df_bounded = df_bounded[df_bounded["ScientificName"].notna()]
    species_counts = df_bounded["ScientificName"].value_counts()
    species_to_keep = species_counts[species_counts >= MIN_OCCURRENCES].index
    df_filtered = df_bounded[df_bounded["ScientificName"].isin(species_to_keep)]

    cutoff = datetime.now().year - YEARS_BACK
    df_recent = df_filtered[df_filtered["YearCollected"] >= cutoff]
    if len(df_recent) > 0:
        df_filtered = df_recent

    coords = df_filtered[[LAT_COL, LON_COL]].to_numpy()
    labels = df_filtered["ScientificName"].to_numpy()

    if len(coords) == 0:
        print("No records after filtering; index not built.")
        return None

    tree = KDTree(coords)
    print(f"Index built: points={len(coords)}")
    return SpeciesIndex(coords=coords, labels=labels, tree=tree)


SPECIES_INDEX = build_index()
INDEX_STATUS = (
    f"Index built: {len(SPECIES_INDEX.coords)} points"
    if SPECIES_INDEX is not None
    else "Warning: No records after filtering; predictions will be unavailable."
)


# distance of 0.1deg -> roughly 7x11km
def predict_plants(lat, lon, max_distance=0.05, top_k=20):
    """Predict plant species likelihood at given coordinates using spatial K-NN."""
    if SPECIES_INDEX is None:
        return None

    indices = SPECIES_INDEX.tree.query_ball_point([lat, lon], r=max_distance)
    if not indices:
        return None

    neighbors = SPECIES_INDEX.labels[indices]
    neighbor_coords = SPECIES_INDEX.coords[indices]

    deltas = neighbor_coords - np.array([lat, lon], dtype=float)
    distances = np.linalg.norm(deltas, axis=1)
    weights = 1 / (distances + 1e-6)

    species_weights = defaultdict(float)
    for species, weight in zip(neighbors, weights):
        species_weights[species] += weight

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
    if lat is None or lon is None:
        return "Invalid input. Please retry."
    predictions = predict_plants(lat=lat, lon=lon)
    return format_predictions(predictions=predictions)


def format_predictions(predictions):
    """Formats predictions for display"""
    if not predictions:
        return "No plants found nearby. Maybe try zooming in?"
    lines = []
    for species, prob in predictions:
        name = species if species.strip() else "(Unknown species)"
        lines.append(f"{name}: {prob:.2f}%")
    return "\n".join(lines)


def predict_from_coord(text: str):
    """Parse 'lat,lon' and run prediction."""
    if not text or "," not in text:
        return "Invalid input. Please retry."
    try:
        lat_s, lon_s = [p.strip() for p in text.split(",", 1)]
        lat = float(lat_s)
        lon = float(lon_s)
    except Exception:
        return "Invalid input. Please retry."
    if not (MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON):
        return f"Out of bounds. Try lat {MIN_LAT}–{MAX_LAT}, lon {MIN_LON}–{MAX_LON}."
    return predict(lat, lon)


with gr.Blocks() as demo:
    gr.Markdown("Click anywhere on the map to predict plant species at that location")
    gr.Markdown(INDEX_STATUS)

    leaflet_html = open("map.html").read()
    gr.HTML(leaflet_html)
    gr.HTML("<style>#coord_input{display:none}</style>")

    coord_input = gr.Textbox(
        label="Coordinates",
        elem_id="coord_input",
        value="",
        lines=1,
        interactive=True,
        visible=True,  # will be hidden by JS
    )

    output = gr.Textbox(
        label="Prediction Output",
        placeholder="Click on the map to get prediction here...",
    )

    coord_input.input(fn=predict_from_coord, inputs=coord_input, outputs=output)

    js_code = open("assets/map_init.js", "r", encoding="utf-8").read()
    demo.load(lambda: None, inputs=[], outputs=[], js=js_code)

demo.launch()
