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


def _neighbors_with_zoom(
    tree: KDTree, point, r0: float, r_max: float, min_count: int, growth: float = 1.8
):
    """Grow the search radius until we find at least min_count neighbors or hit r_max."""
    r = float(r0)
    while r <= r_max:
        idx = tree.query_ball_point(point, r)
        if idx:
            if len(idx) >= min_count or r >= r_max:
                return idx, r
        r *= growth
    return [], r_max


# distance of 0.1deg -> roughly 7x11km
def predict_plants(
    lat, lon, max_distance=0.05, top_k=25, min_neighbors=5, max_radius=1
):
    """Predict plant species likelihood at given coordinates using spatial K-NN."""
    if SPECIES_INDEX is None:
        return None, 0

    indices, r_used = _neighbors_with_zoom(
        SPECIES_INDEX.tree,
        [lat, lon],
        r0=max_distance,
        r_max=max_radius,
        min_count=min_neighbors,
    )
    if not indices:
        return None, r_used

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
        return None, r_used

    results = [
        (species_name, 100 * weight / total_weight)
        for species_name, weight in species_weights.items()
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k], r_used


def predict(lat, lon):
    """Main prediction interface"""
    if lat is None or lon is None:
        return "Invalid input. Please retry."
    predictions, r_used = predict_plants(lat=lat, lon=lon)
    return format_predictions(predictions=predictions, r_used=r_used, lat=lat)


def format_predictions(predictions, r_used, lat=None):
    """Formats predictions for display"""
    LAT_TO_KM = 111  # approx km per degree latitude
    if not predictions:
        return "No plants found nearby."
    lat_km = LAT_TO_KM * r_used
    lon_km = LAT_TO_KM * np.cos(np.radians(lat)) * r_used  # since lon depends on lat
    lines = [f"Approximate area searched: {lon_km*2:.1f} km x {lat_km*2:.1f} km"]
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
        return f"Out of bounds. Please click inside the bounds."
    return predict(lat, lon)


with gr.Blocks() as demo:
    gr.Markdown(
        "Click anywhere on the map to generate an approximate estimate of plant species at that location. Results are based on K-NN on an academic dataset, which may contain inaccuracies."
    )

    gr.HTML("<style>.gradio-container{max-width:100% !important}</style>")
    gr.HTML("<style>#coord_input{display:none}</style>")

    coord_input = gr.Textbox(
        label="Coordinates",
        elem_id="coord_input",
        value="",
        lines=1,
        interactive=True,
        visible=True,
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=3, min_width=420):
            leaflet_html = open("map.html", encoding="utf-8").read()
            gr.HTML(leaflet_html)
        with gr.Column(scale=2, min_width=320):
            output = gr.Textbox(
                label="Prediction Output",
                placeholder="Click on the map to get prediction here...",
                lines=25,
                show_copy_button=True,
            )

    coord_input.input(fn=predict_from_coord, inputs=coord_input, outputs=output)

    js_code = open("assets/map_init.js", "r", encoding="utf-8").read()
    demo.load(lambda: None, inputs=[], outputs=[], js=js_code)

demo.launch()
