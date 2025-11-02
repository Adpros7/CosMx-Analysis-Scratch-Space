"""Python translation of `0.-loading-flat-files.Rmd`.

The workflow follows the CosMx Analysis Scratch Space vignette:
https://nanostring-biostats.github.io/CosMx-Analysis-Scratch-Space/.
It loads CosMx/AtoMx flat files, streams counts into sparse matrices,
separates control probes, and stores an intermediate SpatialData object.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path.cwd()
UTILS_DIR = PROJECT_ROOT / "_code" / "vignette" / "python"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from utils import (  # noqa: E402
    CosMxFlatFileData,
    cosmx_to_spatialdata,
    load_cosmx_flatfiles,
    save_processed_objects,
)

# ---------------------------------------------------------------------------
# 1. Locate CosMx flat files and stream them in chunks
# ---------------------------------------------------------------------------

flatfile_dir = Path("../data/flatFiles")
processed_dir = Path("../processed_data")
chunk_target = int(5e7)  # matches the R vignette default

cosmx_data: CosMxFlatFileData = load_cosmx_flatfiles(flatfile_dir, chunk_nnz_target=chunk_target)

print("Counts matrix shape (cells x genes):", cosmx_data.counts.shape)
print("Negative controls shape:", cosmx_data.negcounts.shape)
print("System controls shape:", cosmx_data.falsecounts.shape)
print("Metadata columns:", cosmx_data.metadata.columns.tolist())

# ---------------------------------------------------------------------------
# 2. Assemble a SpatialData container for downstream analyses
# ---------------------------------------------------------------------------

try:
    spatial_data = cosmx_to_spatialdata(cosmx_data)
    print(spatial_data)
except ImportError:
    spatial_data = None
    print(
        "SpatialData is unavailable. Install `spatialdata` to enable spatial omics plotting "
        "(Gerber et al., Nat. Methods 2023)."
    )

# ---------------------------------------------------------------------------
# 3. Persist intermediary outputs for convenience
# ---------------------------------------------------------------------------

save_processed_objects(
    output_dir=processed_dir,
    counts=cosmx_data.counts,
    metadata=cosmx_data.metadata,
    negcounts=cosmx_data.negcounts,
    falsecounts=cosmx_data.falsecounts,
)

cosmx_data.metadata.to_parquet(processed_dir / "metadata_unfiltered.parquet")
np.savez_compressed(
    processed_dir / "gene_lists.npz",
    genes=cosmx_data.gene_names.to_numpy(),
    negative_controls=cosmx_data.control_names["negative"].to_numpy(),
    system_controls=cosmx_data.control_names["system_control"].to_numpy(),
)

print("Processed objects written to:", processed_dir.resolve())
