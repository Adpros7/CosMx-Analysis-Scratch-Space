"""Python translation of `2. QC and normalization.Rmd`.

This notebook reproduces the CosMx QC workflow: cell-level filters, field-of-
view (FOV) QC using the barcode-aware method from Kennedy et al. (2024), and
global-scaling normalisation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import read_h5ad
from scipy import sparse

PROJECT_ROOT = Path.cwd()
UTILS_DIR = PROJECT_ROOT / "_code" / "vignette" / "python"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from utils import (  # noqa: E402
    load_barcode_map,
    plot_flagged_fovs,
    plot_fov_effects,
    plot_fov_effects_heatmap,
    plot_fov_signal_loss,
    run_fov_qc,
    save_processed_objects,
)

processed_dir = Path("../processed_data")
adata = read_h5ad(processed_dir / "cosmx_processed.h5ad")
metadata = adata.obs.copy()
counts = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
negcounts = adata.layers["negcounts"].tocsr()
falsecounts = adata.layers["falsecounts"].tocsr()
xy_df = pd.read_parquet(processed_dir / "xy_unfiltered.parquet")
xy = xy_df[["x_mm", "y_mm"]].to_numpy()

# ---------------------------------------------------------------------------
# Cell-level QC
# ---------------------------------------------------------------------------

count_threshold = 20
flag = metadata["nCount_RNA"].to_numpy() < count_threshold
print("Cells flagged for low counts:", flag.sum())

plt.figure()
plt.hist(metadata["Area"], bins=100)
plt.axvline(30000, color="red")
plt.xlabel("Cell area")
plt.ylabel("Frequency")
plt.title("Cell area distribution")
plt.show()

area_threshold = 30000
flag = flag | (metadata["Area"].to_numpy() > area_threshold)
print("Cells flagged after area filter:", flag.sum())

# ---------------------------------------------------------------------------
# FOV-level QC
# ---------------------------------------------------------------------------

barcode_rds = PROJECT_ROOT / "_code" / "FOV QC" / "barcodes_by_panel.RDS"
barcode_map = load_barcode_map(barcode_rds, panel="Hs_UCC")

qc_result = run_fov_qc(
    counts=counts,
    gene_names=adata.var_names.tolist(),
    xy=xy,
    fov=metadata["FOV"],
    tissue=metadata["tissue"] if "tissue" in metadata else None,
    barcodemap=barcode_map,
    max_prop_loss=0.6,
    max_totalcounts_loss=0.6,
    n_neighbors=10,
)

plot_flagged_fovs(qc_result)
plot_fov_signal_loss(qc_result)
plot_fov_effects(qc_result, bits="flagged_reportercycles")
plot_fov_effects_heatmap(qc_result)

print("Flagged FOVs (any reason):", qc_result.flagged_fovs)
print("Flagged FOVs for total counts:", qc_result.flagged_fovs_for_total_counts)
print("Flagged FOVs for bias:", qc_result.flagged_fovs_for_bias)
print("Genes affected in flagged FOVs:")
print(qc_result.flagged_fov_gene_pairs.drop_duplicates().head())

flag = flag | metadata["FOV"].isin(qc_result.flagged_fovs).to_numpy()
print("Cells flagged after FOV QC:", flag.sum())

# ---------------------------------------------------------------------------
# Remove flagged cells and save filtered data
# ---------------------------------------------------------------------------

keep_mask = ~flag
counts_filtered = counts[keep_mask]
negcounts_filtered = negcounts[keep_mask]
falsecounts_filtered = falsecounts[keep_mask]
metadata_filtered = metadata.loc[keep_mask].copy()
xy_filtered = xy[keep_mask]

filtered_dir = processed_dir / "filtered"
save_processed_objects(
    output_dir=filtered_dir,
    counts=counts_filtered,
    metadata=metadata_filtered,
    negcounts=negcounts_filtered,
    falsecounts=falsecounts_filtered,
)
metadata_filtered.to_parquet(filtered_dir / "metadata_filtered.parquet")
pd.DataFrame(xy_filtered, columns=["x_mm", "y_mm"], index=metadata_filtered.index).to_parquet(
    filtered_dir / "xy_filtered.parquet"
)

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

scaling_factor = metadata_filtered["nCount_RNA"].mean()
scales = scaling_factor / metadata_filtered["nCount_RNA"].to_numpy()
normalization_matrix = sparse.diags(scales)
norm_counts = normalization_matrix @ counts_filtered

normalized_dir = processed_dir / "normalized"
save_processed_objects(
    output_dir=normalized_dir,
    counts=counts_filtered,
    metadata=metadata_filtered,
    negcounts=negcounts_filtered,
    falsecounts=falsecounts_filtered,
    normalized_counts=norm_counts,
)
print("Normalization complete. Outputs written to:", normalized_dir.resolve())
