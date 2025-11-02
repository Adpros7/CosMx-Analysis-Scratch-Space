"""Utility helpers for the CosMx workflow Python translations.

The implementation mirrors the original R vignettes from the CosMx Analysis
Scratch Space repository, including chunked flat-file loading, the shelf
algorithm for tissue condensation, and the field-of-view (FOV) QC workflow.
Where relevant the functions cite the original Bruker Spatial Biology
resources so notebook users can cross-reference algorithmic details.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.stats import ttest_1samp
from sklearn.neighbors import NearestNeighbors

try:  # pyreadr is required only when barcode maps are read from RDS files
    import pyreadr  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyreadr = None

try:  # SpatialData is optional; raise a clear error if unavailable
    from spatialdata import SpatialData
except ImportError:  # pragma: no cover - optional dependency
    SpatialData = None  # type: ignore


GOOGLE_CATEGORY20 = [
    "#616161", "#4285f4", "#db4437", "#f4b400", "#0f9d58", "#ab47bc",
    "#00acc1", "#ff7043", "#9e9d24", "#5c6bc0", "#f06292", "#00796b",
    "#c2185b", "#7e57c2", "#03a9f4", "#8bc34a", "#fdd835", "#fb8c00",
    "#8d6e63", "#9e9e9e", "#607d8b",
]


@dataclass
class CosMxFlatFileData:
    """Container for CosMx flat file content with control probes separated."""

    counts: csr_matrix
    metadata: pd.DataFrame
    negcounts: csr_matrix
    falsecounts: csr_matrix
    gene_names: pd.Index
    control_names: Dict[str, pd.Index]


@dataclass
class GridInfo:
    """Grid assignments used during the FOV QC computations."""

    grid_ids: np.ndarray
    grid_fov: pd.Series


@dataclass
class FOVQCStatistics:
    """Per-bit summary tables returned by :func:`summarize_fov_bias`."""

    flag: pd.DataFrame
    bias: pd.DataFrame
    p_value: pd.DataFrame
    prop_agree: pd.DataFrame


@dataclass
class FOVQCResult:
    """Result bundle produced by :func:`run_fov_qc`."""

    flagged_fovs: List[str]
    flagged_fovs_for_total_counts: List[str]
    flagged_fovs_for_bias: List[str]
    flagged_fov_gene_pairs: pd.DataFrame
    flags_per_fov_cycle: pd.DataFrame
    fovstats: FOVQCStatistics
    residuals: pd.DataFrame
    total_count_residuals: pd.Series
    grid_info: GridInfo
    xy: pd.DataFrame
    fov_ids: pd.Series
    bitcounts: pd.DataFrame
    gene2bitmap: pd.DataFrame


# ---------------------------------------------------------------------------
# Loading CosMx flat files (0.-loading-flat-files.Rmd)
# ---------------------------------------------------------------------------

def _discover_slides(flatfile_dir: Path) -> List[Path]:
    slides = sorted(p for p in flatfile_dir.iterdir() if p.is_dir())
    if not slides:
        raise FileNotFoundError(
            f"No slide folders detected under {flatfile_dir}. "
            "Confirm that the exported CosMx flat files are present."
        )
    return slides


def _select_file(slide_dir: Path, token: str) -> Path:
    matches = [p for p in slide_dir.iterdir() if token in p.name]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file containing '{token}' in {slide_dir}, found {len(matches)}"
        )
    return matches[0]


def _read_metadata(metadata_file: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_file)
    if "slide_ID" not in metadata.columns:
        raise ValueError("metadata file is missing the 'slide_ID' column")
    return metadata


def _read_counts_sparse(
    counts_file: Path,
    slide_numeric_id: int,
    chunk_nnz_target: int = 50_000_000,
) -> Tuple[csr_matrix, List[str], pd.Index]:
    compression = "gzip" if counts_file.suffix == ".gz" else "infer"
    req_cols = pd.read_csv(counts_file, usecols=["fov", "cell_ID"], compression=compression)
    if not {"fov", "cell_ID"}.issubset(req_cols.columns):
        raise ValueError("Counts matrix must contain 'fov' and 'cell_ID' columns")

    n_cells = req_cols.shape[0]
    sample_chunk = pd.read_csv(counts_file, nrows=1, compression=compression)
    n_cols = sample_chunk.shape[1]
    n_chunks = max(int(math.ceil(n_cells * n_cols / chunk_nnz_target)), 1)
    chunk_size = max(int(math.ceil(n_cells / n_chunks)), 1)

    matrices: List[csr_matrix] = []
    cell_ids: List[str] = []
    genes = sample_chunk.columns[2:]

    reader = pd.read_csv(counts_file, chunksize=chunk_size, compression=compression)
    for chunk in reader:
        if chunk.empty:
            continue
        labels = [
            f"c_{slide_numeric_id}_{fov}_{cell_id}"
            for fov, cell_id in zip(chunk["fov"], chunk["cell_ID"])
        ]
        matrix = sparse.csr_matrix(chunk.iloc[:, 2:].to_numpy(dtype=np.float32))
        matrices.append(matrix)
        cell_ids.extend(labels)

    if not matrices:
        raise ValueError(f"No data read from {counts_file}")

    counts = sparse.vstack(matrices, format="csr")
    return counts, cell_ids, genes


def load_cosmx_flatfiles(
    flatfile_dir: Path,
    chunk_nnz_target: int = 50_000_000,
) -> CosMxFlatFileData:
    """Load CosMx flat files, mirroring the original R vignette implementation."""

    slides = _discover_slides(flatfile_dir)

    count_blocks: List[csr_matrix] = []
    metadata_frames: List[pd.DataFrame] = []
    slide_gene_sets: List[pd.Index] = []
    slide_cell_ids: List[List[str]] = []

    for slide_dir in slides:
        metadata_file = _select_file(slide_dir, "metadata_file")
        counts_file = _select_file(slide_dir, "exprMat_file")

        metadata = _read_metadata(metadata_file)
        slide_numeric_id = int(metadata["slide_ID"].iloc[0])

        counts, cell_ids, genes = _read_counts_sparse(
            counts_file, slide_numeric_id, chunk_nnz_target=chunk_nnz_target
        )

        metadata = metadata.assign(
            cell_global_id=[
                f"c_{slide_numeric_id}_{fov}_{cell_id}"
                for fov, cell_id in zip(metadata["fov"], metadata["cell_ID"])
            ]
        )
        metadata_frames.append(metadata)
        slide_gene_sets.append(pd.Index(genes))
        slide_cell_ids.append(cell_ids)
        count_blocks.append(counts)

    shared_genes = slide_gene_sets[0]
    for idx in slide_gene_sets[1:]:
        shared_genes = shared_genes.intersection(idx)

    aligned_counts: List[csr_matrix] = []
    aligned_metadata: List[pd.DataFrame] = []
    for counts, metadata, cell_ids, gene_idx in zip(
        count_blocks, metadata_frames, slide_cell_ids, slide_gene_sets
    ):
        gene_idx = pd.Index(gene_idx)
        keep_mask = gene_idx.isin(shared_genes)
        counts = counts[:, keep_mask]
        # reorder to match shared_genes ordering (preserves first-slide order)
        reorderer = gene_idx[keep_mask].get_indexer(shared_genes)
        counts = counts[:, reorderer]
        counts = counts.tocsr()

        metadata = metadata.assign(cell_global_id=cell_ids)
        aligned_counts.append(counts)
        aligned_metadata.append(metadata)

    counts_all = sparse.vstack(aligned_counts, format="csr")
    metadata_all = pd.concat(aligned_metadata, ignore_index=True)
    metadata_all = metadata_all.loc[:, ~metadata_all.columns.duplicated()]
    metadata_all = metadata_all.set_index("cell_global_id", drop=False)
    metadata_all["FOV"] = metadata_all.apply(lambda row: f"s{row['slide_ID']}f{row['fov']}", axis=1)

    gene_index = pd.Index(shared_genes, name="gene")
    neg_mask = gene_index.str.contains("Negative", case=False, regex=True)
    false_mask = gene_index.str.contains("SystemControl", case=False, regex=True)

    negcounts = counts_all[:, neg_mask]
    falsecounts = counts_all[:, false_mask]
    real_mask = ~(neg_mask | false_mask)
    real_counts = counts_all[:, real_mask]
    real_gene_index = gene_index[real_mask]

    control_names = {
        "negative": gene_index[neg_mask],
        "system_control": gene_index[false_mask],
    }

    return CosMxFlatFileData(
        counts=real_counts,
        metadata=metadata_all,
        negcounts=negcounts,
        falsecounts=falsecounts,
        gene_names=real_gene_index,
        control_names=control_names,
    )


def cosmx_to_spatialdata(data: CosMxFlatFileData) -> SpatialData:
    """Create a :class:`SpatialData` object (Gerber et al., Nat. Methods 2023)."""

    if SpatialData is None:
        raise ImportError("The 'spatialdata' package is required for this operation")

    adata = AnnData(
        X=data.counts,
        obs=data.metadata.copy(),
        var=pd.DataFrame(index=data.gene_names),
    )
    adata.layers["negcounts"] = data.negcounts
    adata.layers["falsecounts"] = data.falsecounts

    if {"CenterX_global_px", "CenterY_global_px"}.issubset(adata.obs.columns):
        xy = adata.obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy()
        adata.obsm["spatial"] = xy

    return SpatialData(table=adata)


# ---------------------------------------------------------------------------
# Tissue arrangement utilities (1. finessing tissues spatial arrangement)
# ---------------------------------------------------------------------------

def condense_tissues(
    xy: np.ndarray,
    tissue: Sequence[str],
    tissue_order: Optional[Sequence[str]] = None,
    buffer: float = 0.2,
    width_height_ratio: float = 4 / 3,
) -> np.ndarray:
    """Shelf algorithm ported from ``condenseTissues.R``."""

    xy = np.asarray(xy, dtype=float).copy()
    tissue = np.asarray(tissue)
    unique_tissues = pd.unique(tissue)

    dims = []
    for label in unique_tissues:
        mask = tissue == label
        width = float(np.nanmax(xy[mask, 0]) - np.nanmin(xy[mask, 0]))
        height = float(np.nanmax(xy[mask, 1]) - np.nanmin(xy[mask, 1]))
        dims.append((label, width, height))
    dims_df = pd.DataFrame(dims, columns=["tissue", "width", "height"])

    if tissue_order is not None:
        tissue_order = list(tissue_order)
        if set(tissue_order) != set(dims_df["tissue"]):
            raise ValueError("tissue_order must include exactly the unique tissue labels")
        dims_df["order"] = pd.Categorical(dims_df["tissue"], categories=tissue_order, ordered=True)
        dims_df = dims_df.sort_values("order")
    else:
        dims_df = dims_df.sort_values("height", ascending=False)

    n_shelf = max(
        int(round(math.sqrt(len(dims_df)) * width_height_ratio * dims_df["height"].mean() / dims_df["width"].mean())),
        1,
    )
    target_width = (
        dims_df["width"].iloc[:n_shelf].sum() + buffer * max(n_shelf - 1, 0)
        if len(dims_df) >= n_shelf
        else dims_df["width"].sum()
    )

    placements: Dict[str, Tuple[float, float]] = {}
    current_x = 0.0
    current_y = 0.0
    shelf_height = 0.0
    shelf_width = 0.0

    for idx, row in dims_df.reset_index(drop=True).iterrows():
        label = row["tissue"]
        placements[label] = (current_x, current_y)
        shelf_height = max(shelf_height, row["height"])
        shelf_width = current_x + row["width"]
        current_x += row["width"] + buffer

        if idx < len(dims_df) - 1:
            next_width = dims_df.iloc[idx + 1]["width"]
            if abs(shelf_width - target_width) < abs(shelf_width + buffer + next_width - target_width):
                current_y += shelf_height + buffer
                current_x = 0.0
                shelf_height = 0.0
                shelf_width = 0.0

    for label, (offset_x, offset_y) in placements.items():
        mask = tissue == label
        xy[mask, 0] = xy[mask, 0] - np.nanmin(xy[mask, 0]) + offset_x
        xy[mask, 1] = xy[mask, 1] - np.nanmin(xy[mask, 1]) + offset_y

    return xy


# ---------------------------------------------------------------------------
# FOV QC utilities (2. QC and normalization.Rmd & FOV QC utils.R)
# ---------------------------------------------------------------------------

def load_barcode_map(barcode_rds: Path, panel: str) -> pd.DataFrame:
    """Load the gene-to-barcode map for the requested panel."""

    if pyreadr is None:
        raise ImportError("Install 'pyreadr' to read barcode RDS files (see CosMx FOV QC vignette)")
    loaded = pyreadr.read_r(barcode_rds)
    if panel not in loaded:
        raise KeyError(f"Panel '{panel}' not found in {barcode_rds}. Available panels: {list(loaded.keys())}")
    df = loaded[panel]
    df.columns = [col.lower() for col in df.columns]
    if not {"gene", "barcode"}.issubset(df.columns):
        raise ValueError("Barcode map must contain 'gene' and 'barcode' columns")
    df = df.loc[:, ["gene", "barcode"]].dropna()
    df["gene"] = df["gene"].astype(str)
    df["barcode"] = df["barcode"].astype(str)
    return df


def _get_color_values(barcodes: Iterable[str]) -> List[str]:
    return sorted({char for barcode in barcodes for char in barcode if char != "."})


def barcode_to_bit_matrix(barcodes: Sequence[str]) -> pd.DataFrame:
    if not barcodes:
        raise ValueError("barcode list is empty")
    barcode_len = len(barcodes[0])
    if any(len(bc) != barcode_len for bc in barcodes):
        raise ValueError("All barcodes must have the same length")

    colors = _get_color_values(barcodes)
    n_cycles = barcode_len // 2
    bit_names = [
        f"reportercycle{cycle}{color}"
        for cycle in range(1, n_cycles + 1)
        for color in colors
    ]

    rows: List[List[int]] = []
    for barcode in barcodes:
        row: List[int] = []
        for cycle in range(1, n_cycles + 1):
            color_here = barcode[cycle * 2 - 1]
            for candidate in colors:
                row.append(1 if color_here == candidate else 0)
        rows.append(row)

    return pd.DataFrame(rows, columns=bit_names, dtype=int)


def make_grid(
    xy: np.ndarray,
    fov: Sequence[str],
    squares_per_fov: int = 49,
    min_cells_per_square: int = 25,
) -> GridInfo:
    xy = np.asarray(xy)
    fov = np.asarray(fov)
    ncuts = int(np.floor(np.sqrt(squares_per_fov)))
    if ncuts < 1:
        raise ValueError("squares_per_fov must be at least 1")

    grid_ids = np.full(len(fov), None, dtype=object)
    grid_fov_map: Dict[str, str] = {}

    for fov_id in pd.unique(fov):
        mask = fov == fov_id
        if mask.sum() == 0:
            continue
        x_bins = pd.cut(xy[mask, 0], bins=ncuts, labels=False)
        y_bins = pd.cut(xy[mask, 1], bins=ncuts, labels=False)
        combined = [f"{fov_id}_{x}_{y}" for x, y in zip(x_bins, y_bins)]
        grid_ids[mask] = combined
        for grid in pd.unique(combined):
            grid_fov_map[str(grid)] = fov_id

    counts = pd.Series(grid_ids).value_counts()
    too_small = counts[counts < min_cells_per_square].index
    mask_small = pd.Series(grid_ids).isin(too_small)
    grid_ids[mask_small.to_numpy()] = None

    grid_fov_series = pd.Series(grid_fov_map)
    return GridInfo(grid_ids=grid_ids, grid_fov=grid_fov_series)


def cell_gene_to_grid_bit(
    counts: csr_matrix,
    grid_ids: Sequence[Optional[str]],
    gene_names: Sequence[str],
    barcodemap: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gene_index = pd.Index(gene_names, name="gene")
    barcode_df = barcodemap.drop_duplicates(subset="gene").set_index("gene")
    shared_genes = gene_index.intersection(barcode_df.index)
    if shared_genes.empty:
        raise ValueError("No overlap between counts genes and barcode map genes")

    col_pos = gene_index.get_indexer(shared_genes)
    counts_subset = counts[:, col_pos]

    grid_ids = np.asarray(grid_ids)
    valid_mask = pd.notna(grid_ids)
    if valid_mask.sum() == 0:
        raise ValueError("No grid squares retained after filtering")

    counts_subset = counts_subset[valid_mask]
    grid_ids_valid = grid_ids[valid_mask].astype(str)
    unique_grids = pd.Index(grid_ids_valid).unique()
    grid_indexer = pd.Series(np.arange(len(unique_grids)), index=unique_grids)
    row_ind = grid_indexer.loc[grid_ids_valid].to_numpy()
    col_ind = np.arange(len(grid_ids_valid))
    grid_map = sparse.csr_matrix(
        (np.ones_like(col_ind, dtype=float), (row_ind, col_ind)),
        shape=(len(unique_grids), len(grid_ids_valid)),
    )

    grid_gene = grid_map @ counts_subset
    ncells = np.array(grid_map.sum(axis=1)).ravel()
    ncells[ncells == 0] = 1
    inv = sparse.diags(1.0 / ncells)
    grid_gene = inv @ grid_gene

    grid_gene_df = pd.DataFrame(
        grid_gene.toarray(), index=unique_grids, columns=shared_genes
    )

    barcodes = barcode_df.loc[shared_genes, "barcode"].tolist()
    gene2bitmap = barcode_to_bit_matrix(barcodes)
    gene2bitmap.index = shared_genes

    grid_bit = grid_gene_df @ gene2bitmap
    return grid_bit, gene2bitmap


def get_nearest_neighbors_by_fov(
    x: pd.DataFrame,
    grid_fov: pd.Series,
    n_neighbors: int = 10,
) -> np.ndarray:
    if x.shape[0] < 2:
        raise ValueError("At least two grid squares are required for neighbour matching")

    max_k = min(n_neighbors * 50, x.shape[0])
    nbrs = NearestNeighbors(n_neighbors=max_k).fit(x)
    _, indices = nbrs.kneighbors(x)

    grid_fov = grid_fov.loc[x.index]
    selections = np.full((x.shape[0], n_neighbors), np.nan)

    for i in range(x.shape[0]):
        current_fov = grid_fov.iloc[i]
        chosen: List[int] = []
        seen_fovs: set = set()
        for idx in indices[i]:
            if idx == i:
                continue
            candidate_fov = grid_fov.iloc[idx]
            if candidate_fov == current_fov:
                continue
            if candidate_fov in seen_fovs:
                continue
            chosen.append(idx)
            seen_fovs.add(candidate_fov)
            if len(chosen) == n_neighbors:
                break
        selections[i, : len(chosen)] = chosen

    return selections


def summarize_fov_bias(
    residuals: pd.DataFrame,
    grid_fov: pd.Series,
    max_prop_loss: float,
) -> FOVQCStatistics:
    grid_fov = grid_fov.loc[residuals.index]

    bias_rows = []
    pvalue_rows = []
    prop_rows = []
    threshold = np.log2(1 - max_prop_loss) / 3.0

    for fov_id, df in residuals.groupby(grid_fov):
        bias_rows.append(pd.Series(df.mean(axis=0), name=fov_id))
        prop_rows.append(pd.Series((df < threshold).mean(axis=0), name=fov_id))
        if df.shape[0] < 2:
            pvalue_rows.append(pd.Series(1.0, index=df.columns, name=fov_id))
        else:
            _, p_vals = ttest_1samp(df, popmean=0.0, axis=0, nan_policy="omit")
            if np.isscalar(p_vals):
                p_vals = np.repeat(p_vals, df.shape[1])
            pvalue_rows.append(pd.Series(p_vals, index=df.columns, name=fov_id))

    bias = pd.DataFrame(bias_rows)
    prop_agree = pd.DataFrame(prop_rows)
    p_value = pd.DataFrame(pvalue_rows)

    flag = (bias < np.log2(1 - max_prop_loss)) & (p_value < 0.01) & (prop_agree >= 0.5)

    return FOVQCStatistics(
        flag=flag.astype(float),
        bias=bias,
        p_value=p_value,
        prop_agree=prop_agree,
    )


def run_fov_qc(
    counts: csr_matrix,
    gene_names: Sequence[str],
    xy: np.ndarray,
    fov: Sequence[str],
    barcodemap: pd.DataFrame,
    tissue: Optional[Sequence[str]] = None,
    max_prop_loss: float = 0.6,
    max_totalcounts_loss: float = 0.6,
    n_neighbors: int = 10,
) -> FOVQCResult:
    if not 0 <= max_prop_loss <= 1:
        raise ValueError("max_prop_loss must be between 0 and 1")

    if tissue is None:
        tissue = ["tissue"] * len(fov)
    fused_fov = [f"{t}_{fv}" for t, fv in zip(tissue, fov)]

    grid_info = make_grid(xy, fused_fov, squares_per_fov=49, min_cells_per_square=10)
    grid_bit, gene2bitmap = cell_gene_to_grid_bit(
        counts=counts,
        grid_ids=grid_info.grid_ids,
        gene_names=gene_names,
        barcodemap=barcodemap,
    )

    grid_bit = grid_bit.div(grid_bit.sum(axis=1), axis=0).mul(grid_bit.sum(axis=1).mean())
    comparators = get_nearest_neighbors_by_fov(grid_bit, grid_info.grid_fov, n_neighbors=n_neighbors)

    total_counts = np.asarray(counts.sum(axis=1)).ravel()
    grid_series = pd.Series(grid_info.grid_ids, name="grid")
    total_counts_df = pd.DataFrame({"grid": grid_series, "total": total_counts})
    total_counts_grid = (
        total_counts_df.dropna().groupby("grid")["total"].mean().reindex(grid_bit.index)
    )

    def _mean_neighbors(values: pd.Series) -> np.ndarray:
        arr = values.to_numpy()
        estimates = []
        for neigh in comparators:
            valid = neigh[~np.isnan(neigh)].astype(int)
            if len(valid) == 0:
                estimates.append(np.nan)
            else:
                estimates.append(np.nanmean(arr[valid]))
        return np.array(estimates)

    total_counts_hat = _mean_neighbors(total_counts_grid.fillna(total_counts_grid.mean()))
    total_count_residuals = np.log2((total_counts_grid + 1) / (total_counts_hat + 1))
    total_threshold = np.log2(1 - max_totalcounts_loss)
    flagged_grids = total_count_residuals < total_threshold

    grid_fov_series = grid_info.grid_fov.reindex(grid_bit.index)
    flagged_series = pd.Series(flagged_grids, index=grid_bit.index).fillna(False)
    flags_per_fov_total = (
        flagged_series.astype(float).groupby(grid_fov_series).mean().fillna(0.0)
    )
    flagged_fovs_total = flags_per_fov_total[flags_per_fov_total > 0.75].index.tolist()

    grid_values = grid_bit.to_numpy()
    expectations = []
    for neigh in comparators:
        valid = neigh[~np.isnan(neigh)].astype(int)
        if len(valid) == 0:
            expectations.append(np.zeros(grid_bit.shape[1]))
        else:
            expectations.append(grid_values[valid].mean(axis=0))
    yhat = np.vstack(expectations)
    residuals = np.log2((grid_values + 1) / (yhat + 1))
    residuals_df = pd.DataFrame(residuals, index=grid_bit.index, columns=grid_bit.columns)

    fovstats = summarize_fov_bias(residuals_df, grid_info.grid_fov, max_prop_loss)

    bitnames = fovstats.flag.columns
    cycle_labels = [re.match(r"(reportercycle\d+)", name).group(1) for name in bitnames]
    flags_per_cycle = {}
    for cycle in pd.unique(cycle_labels):
        mask = [label == cycle for label in cycle_labels]
        flags_per_cycle[cycle] = fovstats.flag.loc[:, mask].mean(axis=1)
    flags_per_cycle_df = pd.DataFrame(flags_per_cycle)

    flagged_fovs_bias = flags_per_cycle_df.index[(flags_per_cycle_df >= 0.5).any(axis=1)].tolist()
    flagged_fovs = sorted(set(flagged_fovs_total).union(flagged_fovs_bias))

    flagged_pairs: List[Tuple[str, str]] = []
    for fov_id in flagged_fovs_bias:
        cycles = flags_per_cycle_df.columns[flags_per_cycle_df.loc[fov_id] >= 0.5]
        for cycle in cycles:
            matching_bits = [col for col in bitnames if col.startswith(cycle)]
            affected_genes = gene2bitmap.index[gene2bitmap[matching_bits].sum(axis=1) > 0]
            flagged_pairs.extend([(fov_id, gene) for gene in affected_genes])
    flagged_pairs_df = pd.DataFrame(flagged_pairs, columns=["fov", "gene"]).drop_duplicates()

    xy_df = pd.DataFrame(xy, columns=["x", "y"])
    fov_series = pd.Series(fused_fov, name="fov")

    return FOVQCResult(
        flagged_fovs=flagged_fovs,
        flagged_fovs_for_total_counts=flagged_fovs_total,
        flagged_fovs_for_bias=flagged_fovs_bias,
        flagged_fov_gene_pairs=flagged_pairs_df,
        flags_per_fov_cycle=flags_per_cycle_df,
        fovstats=fovstats,
        residuals=residuals_df,
        total_count_residuals=total_count_residuals,
        grid_info=grid_info,
        xy=xy_df,
        fov_ids=fov_series,
        bitcounts=grid_bit,
        gene2bitmap=gene2bitmap,
    )


# ---------------------------------------------------------------------------
# Plotting helpers mimicking the base R figures
# ---------------------------------------------------------------------------

def _ensure_axes(ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_tissue_layout(
    xy: np.ndarray,
    labels: Sequence[str],
    sample_fraction: float = 0.05,
    ax: Optional[plt.Axes] = None,
    palette: Sequence[str] = GOOGLE_CATEGORY20,
    title: Optional[str] = None,
    seed: int = 1,
) -> plt.Axes:
    ax = _ensure_axes(ax)
    xy = np.asarray(xy)
    labels = np.asarray(labels)

    n = xy.shape[0]
    sample_size = max(int(round(n * sample_fraction)), 1)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=sample_size, replace=False)

    cat = pd.Categorical(labels[indices])
    colors = [palette[idx % len(palette)] for idx in cat.codes]

    ax.scatter(xy[indices, 0], xy[indices, 1], c=colors, s=5, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    if title:
        ax.set_title(title)

    for label in pd.unique(labels):
        mask = labels == label
        ax.text(
            float(np.nanmedian(xy[mask, 0])),
            float(np.nanmax(xy[mask, 1])),
            str(label),
            fontsize=8,
            ha="center",
            va="bottom",
        )
    return ax


def plot_flagged_fovs(result: FOVQCResult, ax: Optional[plt.Axes] = None) -> plt.Axes:
    ax = _ensure_axes(ax)
    xy = result.xy.to_numpy()
    fov_ids = result.fov_ids.to_numpy()

    ax.scatter(xy[:, 0], xy[:, 1], s=2, color="#cccccc")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Flagged FOVs")

    for fov in pd.unique(fov_ids):
        mask = fov_ids == fov
        ax.add_patch(
            Rectangle(
                (xy[mask, 0].min(), xy[mask, 1].min()),
                xy[mask, 0].max() - xy[mask, 0].min(),
                xy[mask, 1].max() - xy[mask, 1].min(),
                facecolor=(0.12, 0.56, 1.0, 0.3),
                edgecolor="none",
            )
        )

    for fov in result.flagged_fovs:
        mask = fov_ids == fov
        ax.add_patch(
            Rectangle(
                (xy[mask, 0].min(), xy[mask, 1].min()),
                xy[mask, 0].max() - xy[mask, 0].min(),
                xy[mask, 1].max() - xy[mask, 1].min(),
                facecolor=(1.0, 0.0, 0.0, 0.3),
                edgecolor="none",
            )
        )
        ax.text(
            float((xy[mask, 0].min() + xy[mask, 0].max()) / 2),
            float((xy[mask, 1].min() + xy[mask, 1].max()) / 2),
            fov,
            color="green",
            fontsize=8,
            ha="center",
            va="center",
        )
    return ax


def plot_fov_signal_loss(result: FOVQCResult, ax: Optional[plt.Axes] = None) -> plt.Axes:
    ax = _ensure_axes(ax)
    xy = result.xy.to_numpy()
    grid_series = pd.Series(result.grid_info.grid_ids)
    color_values = grid_series.map(result.total_count_residuals)
    cmap = LinearSegmentedColormap.from_list(
        "signal_loss",
        ["darkblue", "blue", "lightgrey", "red", "darkred"],
    )

    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=color_values.to_numpy(),
        cmap=cmap,
        s=5,
        vmin=-2,
        vmax=2,
    )
    ax.set_aspect("equal")
    ax.set_title("Log2 fold-change in total counts")
    ax.figure.colorbar(sc, ax=ax, label="log2 fold-change")
    return ax


def plot_fov_effects(
    result: FOVQCResult,
    bits: str = "flagged_reportercycles",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    if bits not in {"flagged_reportercycles", "flagged_bits", "all"}:
        raise ValueError("bits must be 'flagged_reportercycles', 'flagged_bits', or 'all'")

    flagged_cycles = result.flags_per_fov_cycle.columns[
        (result.flags_per_fov_cycle >= 0.5).any(axis=0)
    ]
    if bits == "flagged_reportercycles" and len(flagged_cycles) == 0:
        flagged_cycles = result.flags_per_fov_cycle.columns

    if bits == "flagged_reportercycles":
        bit_names = [
            col
            for cycle in flagged_cycles
            for col in result.residuals.columns
            if col.startswith(cycle)
        ]
    elif bits == "flagged_bits":
        bit_names = result.fovstats.flag.columns[result.fovstats.flag.sum(axis=0) > 0]
    else:
        bit_names = list(result.residuals.columns)

    if not bit_names:
        raise ValueError("No bits available for plotting under the requested mode")

    n_bits = len(bit_names)
    n_cols = 2
    n_rows = math.ceil(n_bits / n_cols)
    figsize = figsize or (6 * n_cols, 6 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    cmap = LinearSegmentedColormap.from_list(
        "bit_bias",
        ["darkblue", "blue", "lightgrey", "red", "darkred"],
    )

    grid_series = pd.Series(result.grid_info.grid_ids)
    xy = result.xy.to_numpy()

    for ax, bit in zip(axes.flat, bit_names):
        color_values = grid_series.map(result.residuals[bit])
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=color_values.to_numpy(),
            cmap=cmap,
            s=5,
            vmin=-1,
            vmax=1,
        )
        ax.set_title(bit)
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax)

    for ax in axes.flat[n_bits:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_fov_effects_heatmap(result: FOVQCResult, ax: Optional[plt.Axes] = None) -> plt.Axes:
    ax = _ensure_axes(ax)
    sns.heatmap(
        result.fovstats.bias * result.fovstats.flag,
        cmap=sns.color_palette(["darkblue", "blue", "white", "red", "darkred"], as_cmap=True),
        vmin=-2,
        vmax=2,
        ax=ax,
    )
    ax.set_title("FOV bias: log2 fold-change from comparable regions")
    return ax


# ---------------------------------------------------------------------------
# Convenience for exporting processed objects
# ---------------------------------------------------------------------------

def save_processed_objects(
    output_dir: Path,
    counts: csr_matrix,
    metadata: pd.DataFrame,
    negcounts: csr_matrix,
    falsecounts: csr_matrix,
    normalized_counts: Optional[csr_matrix] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = AnnData(
        X=counts,
        obs=metadata,
        var=pd.DataFrame(index=np.arange(counts.shape[1])),
        layers={
            "negcounts": negcounts,
            "falsecounts": falsecounts,
        },
    )
    if normalized_counts is not None:
        adata.layers["normalized"] = normalized_counts
    adata.write_h5ad(output_dir / "cosmx_processed.h5ad")
