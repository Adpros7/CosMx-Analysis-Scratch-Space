"""Python translation of `1. finessing tissues spatial arrangement.Rmd`.

The notebook rearranges tissues manually following the CosMx Analysis Scratch
Space tutorial, using the shelf algorithm ported to Python in `utils.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
UTILS_DIR = PROJECT_ROOT / "_code" / "vignette" / "python"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from utils import condense_tissues, plot_tissue_layout  # noqa: E402

processed_dir = Path("../processed_data")
metadata_path = processed_dir / "metadata_unfiltered.parquet"
xy_path = processed_dir / "xy_unfiltered.parquet"

metadata = pd.read_parquet(metadata_path)
xy = metadata[["CenterX_global_px", "CenterY_global_px"]].to_numpy()

instrument_nm_per_pixel = 120.280945  # replace with RunSummary value for your instrument
xy_mm = xy * instrument_nm_per_pixel / 1_000_000.0
xy_mm = xy_mm.astype(float)

plot_tissue_layout(xy_mm, metadata["Run_Tissue_name"], title="Original layout by slide")

xy_condensed = condense_tissues(
    xy=xy_mm,
    tissue=metadata["Run_Tissue_name"],
    tissue_order=None,
    buffer=1.0,
    width_height_ratio=4 / 3,
)
plot_tissue_layout(xy_condensed, metadata["Run_Tissue_name"], title="Condensed by slide")

metadata = metadata.copy()
metadata["tissue"] = pd.NA
metadata.loc[
    (metadata["Run_Tissue_name"] == "MAR19_SlideAge_SkinCancer_RT_CP_slide2")
    & (xy_condensed[:, 1] < 9),
    "tissue",
] = "sample1"
metadata.loc[
    (metadata["Run_Tissue_name"] == "MAR19_SlideAge_SkinCancer_RT_CP_slide2")
    & (xy_condensed[:, 1] >= 9),
    "tissue",
] = "sample2"
metadata.loc[
    (metadata["Run_Tissue_name"] == "MAR19_SlideAge_SkinCancer_4C_CP_slide4")
    & (xy_condensed[:, 1] < 5.3),
    "tissue",
] = "sample3"
metadata.loc[
    (metadata["Run_Tissue_name"] == "MAR19_SlideAge_SkinCancer_4C_CP_slide4")
    & (xy_condensed[:, 1] >= 5.3),
    "tissue",
] = "sample4"

if metadata["tissue"].isna().any():
    raise ValueError("Some cells did not receive a tissue label; adjust thresholds accordingly.")

plot_tissue_layout(xy_condensed, metadata["tissue"], title="Condensed by manual tissue IDs")

xy_tissue_condensed = condense_tissues(
    xy=xy_condensed,
    tissue=metadata["tissue"],
    tissue_order=None,
    buffer=1.0,
    width_height_ratio=1.0,
)
plot_tissue_layout(xy_tissue_condensed, metadata["tissue"], title="Condensed by tissue (square layout)")

sample3_mask = (metadata["tissue"] == "sample3") & (xy_tissue_condensed[:, 1] > 11)
xy_tissue_condensed[sample3_mask, 1] -= 2
plot_tissue_layout(xy_tissue_condensed, metadata["tissue"], title="After manual consolidation")

xy_out = pd.DataFrame(xy_tissue_condensed, columns=["x_mm", "y_mm"], index=metadata.index)
xy_out.to_parquet(xy_path)
metadata.to_parquet(metadata_path)

print("Updated coordinates and metadata saved to:", xy_path.resolve())
