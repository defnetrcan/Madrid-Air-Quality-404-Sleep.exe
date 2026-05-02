# Madrid Air Quality Analytics Pipeline

**Advanced Coding for Data Analytics (2025/2026) — LUISS Guido Carli**

A complete analytics pipeline for the [METRAQ Air Quality Dataset](https://huggingface.co/datasets/dmariaa70/METRAQ-Air-Quality) (Madrid, 2001–2024).

---

## Requirements

Python 3.10+ is required.

Install all dependencies with:

```bash
pip install -r requirements.txt
```

`pyarrow` is optional but recommended — it speeds up the Task 8 partitioning step significantly:

```bash
pip install pyarrow
```

---

## Dataset

Download the full dataset from Hugging Face:

```
https://huggingface.co/datasets/dmariaa70/METRAQ-Air-Quality
```

The pipeline expects a folder containing the yearly CSV files named:

```
metraq_aq-2001.csv
metraq_aq-2002.csv
...
metraq_aq-2024.csv
```

A sample CSV (~100,000 rows) is also supported for quick testing — pass the file path directly instead of the folder.

---

## How to Run

```bash
python madrid.py /path/to/METRAQ-Air-Quality/
```

To run on the sample file:

```bash
python madrid.py sample_madrid_air_quality.csv
```

The script runs all 10 tasks sequentially in a single execution. No flags or subcommands are needed.

---

## Output

All outputs are written automatically to subdirectories created in the working directory:

| Directory | Contents |
|---|---|
| `figures/` | All plots and charts (PNG, 150 dpi) |
| `corr_matrices/` | Per-year, per-sensor correlation matrices (Task 8) |
| `partitions/` | Yearly data partitions used by parallel workers (Task 8) |

The `figures/` folder also contains `task3_best_imputation_methods.csv`, a summary of the best imputation method selected per pollutant.

---

## Configuration

Key constants at the top of `madrid.py` can be adjusted before running:

| Constant | Default | Description |
|---|---|---|
| `SELECTED_POLLUTANTS` | `["NO2", "O3", "<PM10", "SO2"]` | Pollutants used across Tasks 3–10 |
| `KNN_K` | `3` | Number of neighbours for spatial KNN imputation |
| `CORR_THR` | `0.60` | Correlation threshold for network edges (Tasks 6, 8) |
| `TASK8_MAX_WORKERS` | `2` | Parallel worker count — set to your physical core count |
| `TASK8_CORR_MIN_PERIODS` | `168` | Minimum overlapping hours required before computing a correlation |
| `TASK3_EVAL_MASK_FRAC` | `0.03` | Fraction of values masked for imputation evaluation |
| `USE_PARTITIONS` | `True` | Pre-split data into yearly partitions before Task 8 |

---

## What Each Task Does

| Task | Description |
|---|---|
| 1 | Load dataset, inspect schema, compute descriptive statistics, produce distribution plots |
| 2 | Reconstruct original missingness from `is_interpolated`, detect invalid values, analyse temporal and sensor-specific gaps |
| 3 | Compare imputation methods (spatial KNN, rolling mean, ffill/bfill), evaluate via pseudo-gap RMSE, select best causal method per pollutant |
| 4 | Temporal analysis — yearly trends, monthly seasonality, diurnal profiles, year×month heatmaps |
| 5 | Build spatial network (KNN k=2) from UTM coordinates, compute graph metrics, detect communities |
| 6 | Build correlation network from sensor time-series similarity, compare against spatial network |
| 7 *(optional)* | Graph Laplacian diffusion model for NO2 propagation across the sensor network |
| 8 | Parallelised per-year, per-sensor correlation matrices; sequential vs parallel runtime comparison |
| 9 *(optional)* | 24-hour-ahead NO2 forecasting with Ridge regression and Random Forest using meteorological and traffic features |
| 10 | Final summary panels combining key results from all tasks |

---

## Expected Runtime (full dataset, 2 CPU cores, SSD)

| Task | Approx. time |
|---|---|
| 1 — Load + Inspect | 5–10 min |
| 2 — Missingness | 15–20 min |
| 3 — Imputation | 25–40 min |
| 4 — Temporal | 5–10 min |
| 5 — Spatial network | < 1 min |
| 6 — Correlation network | 3–5 min |
| 7 — Propagation model | 10–15 min |
| 8 — Parallelisation | ~15 min |
| 9 — Forecasting | 5–8 min |
| 10 — Final plots | < 1 min |
| **Total** | **~60–90 min** |

Peak RAM usage is approximately 2 GB when loading the full 64 M-row dataset.

---

## Repository Structure

```
.
├── madrid.py                        # Main pipeline script
├── requirements.txt
├── README.md
├── sample_madrid_air_quality.csv    # Small sample for testing
└── figures/                         # Created on first run
```

---

## Citation

David María-Arribas et al. *METRAQ Air Quality dataset.* Hugging Face, 2024.
https://huggingface.co/datasets/dmariaa70/METRAQ-Air-Quality
