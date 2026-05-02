# Madrid Air Quality Analytics Pipeline
## Advanced Coding for Data Analytics (2025/2026) - LUISS Guido Carli

> **Dataset:** METRAQ Air Quality Dataset (Madrid) - 64,881,744 hourly rows · 24 monitoring stations · January 2001 - December 2024  
> **Source:** David María-Arribas et al. - https://huggingface.co/datasets/dmariaa70/METRAQ-Air-Quality

---

## Table of Contents

1. [Project Overview & Goals](#1-project-overview--goals)
2. [Dataset Description](#2-dataset-description)
3. [How to Run](#3-how-to-run)
4. [Repository Structure](#4-repository-structure)
5. [Task 1 - Load Data & Inspect Structure](#5-task-1--load-data--inspect-structure)
6. [Task 2 - Missingness & Data Quality](#6-task-2--missingness--data-quality)
7. [Task 3 - Imputation](#7-task-3--imputation)
8. [Task 4 - Temporal Analysis](#8-task-4--temporal-analysis)
9. [Task 5 - Spatial Network](#9-task-5--spatial-network)
10. [Task 6 - Correlation Network](#10-task-6--correlation-network)
11. [Task 7 - Propagation Modeling *(Optional)*](#11-task-7--propagation-modeling-optional)
12. [Task 8 - Parallelization](#12-task-8--parallelization)
13. [Task 9 - Forecasting Model *(Optional)*](#13-task-9--forecasting-model-optional)
14. [Task 10 - Final Visualization](#14-task-10--final-visualization)
15. [Global Design Decisions & Constants](#15-global-design-decisions--constants)
16. [Scalability & Reproducibility](#16-scalability--reproducibility)

---

## 1. Project Overview & Goals

Air quality is a major public-health concern across European cities. Exposure to pollutants such as NO₂, PM10, and SO₂ is linked to respiratory and cardiovascular disease, making evidence-based monitoring and analysis essential for city planners and health authorities.

This project implements a **complete analytics pipeline** for Madrid air quality using the METRAQ dataset. The pipeline is designed to answer three overarching questions from the project brief:

1. **What are the current pollution patterns in Madrid?** - addressed through temporal analysis (Task 4), spatial analysis (Task 5), and final visualizations (Task 10).
2. **Which areas are most affected?** - addressed through the spatial network and community detection (Task 5), per-sensor correlation analysis (Task 6), and the diffusion model's per-sensor error map (Task 7).
3. **What are the relationships between pollution and other variables (weather, traffic)?** - addressed through correlation network analysis (Task 6), parallelized correlation matrices (Task 8), and the forecasting model (Task 9).

The pipeline covers **all mandatory tasks (1-6, 8, 10)** plus both optional tasks (7 and 9). Every result is fully reproducible by running a single Python script.

---

## 2. Dataset Description

### Schema

Every row represents one hourly measurement at one sensor:

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Column</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Type</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Description</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`sensor_id`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">int32</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Unique numeric ID of the monitoring station</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`sensor_name`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">category</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Human-readable station name (e.g., "Plaza de España")</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`utm_x`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">float32</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">UTM easting coordinate (metres, ETRS89 zone 30N)</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`utm_y`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">float32</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">UTM northing coordinate</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`magnitude_id`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">int16</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Numeric code for the measured variable</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`magnitude_name`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">category</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Variable name (e.g., "NO2", "TEMP", "TI_IDW")</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`entry_date`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">datetime</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Timestamp of the measurement (hourly resolution)</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`value`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">float32</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Observed or interpolated value</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`is_interpolated`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">bool</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`True` = originally missing, filled by METRAQ; `False` = real measurement</td>
  </tr>
  </tbody>
</table>
### Variable Categories

The 36 distinct variables are classified into three categories in `classify_variable()`:

**Air Quality (14 pollutants):**
`<PM10`, `<PM2.5`, `BENCENO`, `CO`, `ETILBENCENO`, `HIDROCARBS_NO_METANICOS`, `HIDROCARBS_TOTALES`, `METANO`, `NO`, `NO2`, `NOX`, `O3`, `SO2`, `TOLUENO`

**Meteorology (7 variables):**
`DV` (wind direction), `HR` (relative humidity), `PRE` (atmospheric pressure), `PRECIPITACION` (rainfall), `RS` (solar radiation), `TEMP` (air temperature), `VV` (wind speed)

**Traffic (15 variables - 3 dimensions × 5 interpolation methods):**
`TI_*` = traffic intensity, `SP_*` = average speed, `OC_*` = road occupancy/congestion.
Suffixes: `IDW`, `KRIGING`, `RBF_GAUSSIAN`, `RBF_LINEAR`, `RBF_MULTIQUADRIQ`.

### Dataset Size

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Category</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Rows</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Percentage</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Traffic</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">30,507,768</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">47.0%</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Air quality</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">25,535,832</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">39.4%</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Meteorology</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">8,838,144</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">13.6%</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">**Total**</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">**64,881,744**</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">100%</td>
  </tr>
  </tbody>
</table>
### Critical Note on Variable Availability

Not all 36 variables span the full 24-year period:

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Variable group</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Available from</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Available to</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Air quality (NO2, NO, NOX, O3, etc.)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">2001-01-01</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">2024-12-31</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">PM2.5</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">2003-01-01</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">2024-12-31</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">METANO, HIDROCARBS</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">2001-01-01</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">2020-12-31</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Traffic (all 15 variables)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">2015-01-01</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">2023-2024</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">**Meteorology (all 7 variables)**</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">**2019-01-01**</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">**2024-12-31**</td>
  </tr>
  </tbody>
</table>
This is why the **forecasting model (Task 9) is constrained to 2019-2024**: meteorological features are required as predictors but only available from 2019.

---

## 3. How to Run

### Requirements

```bash
pip install numpy pandas matplotlib seaborn networkx scipy scikit-learn
pip install pyarrow  # optional, for faster Task 8 partitioning
```

### Execution

```bash
python madrid.py /path/to/METRAQ-Air-Quality/
```

Accepts either a **directory** of yearly CSVs (`metraq_aq-2001.csv` … `metraq_aq-2024.csv`) or a **single CSV** (sample mode). Auto-creates `figures/`, `corr_matrices/`, and `partitions/`.

### Key Configuration Constants

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Constant</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Value</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Justification</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`SELECTED_POLLUTANTS`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`["NO2","O3","<PM10","SO2"]`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">EU Air Quality Directive primary pollutants; full 24-year coverage</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`KNN_K`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`3`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Balances noise reduction vs. geographic specificity</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`CORR_THR`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`0.60`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Standard threshold;</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">r</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">²>0.36 = >36% shared variance</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`TASK3_EVAL_MAX_SENSORS`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`8`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Limits pseudo-gap evaluation cost; top-8 data-rich sensors</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`TASK3_EVAL_N_SEEDS`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`2`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Averages over 2 random masks to reduce RMSE variance</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`TASK3_EVAL_MASK_FRAC`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`0.03`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Masks 3% of observed values - enough to measure fit</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`TASK8_MAX_WORKERS`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`2`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Set to physical core count; adjust for your machine</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`TASK8_CORR_MIN_PERIODS`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`168`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">≥ 1 week hourly data per variable pair before trusting correlation</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`USE_PARTITIONS`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">`True`</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Each Task 8 worker reads only its year's file, not the full 64M rows</td>
  </tr>
  </tbody>
</table>
---

## 4. Repository Structure

```
madrid.py                            # Single script implementing all 10 tasks
README.md                            # This file
figures/                             # All output plots + one CSV summary
corr_matrices/                       # 471 per-year/sensor correlation CSVs
partitions/                          # Year-split data files for Task 8 I/O
```

---

## 5. Task 1 - Load Data & Inspect Structure

### Goal

Understand schema, scale, time coverage, and statistical properties before any analysis. This prevents errors such as using a variable with only 3 years of data in a 24-year trend, or treating traffic RBF-Gaussian values (which collapse near zero) as physically meaningful.

### Why Memory-Efficient dtypes

Default Pandas inference uses float64/object everywhere. By specifying `DTYPES` at load time (`int32` for IDs, `category` for string columns, `float32` for coordinates and values), the in-memory footprint drops ~4×. At 64M rows this is the difference between ~4 GB and ~16 GB RAM - critical on a workstation. `category` dtype is especially efficient for `sensor_name` (24 unique values in 64M rows) and `magnitude_name` (36 unique values).

### Terminal Results

```
[load_dataset] Directory mode: found 24 yearly files
Concatenated 64,881,744 rows from 24 files.

Shape: (64,881,744, 10)
Sensors: 24  |  Variables: 36

Category counts:
  traffic        30,507,768
  air_quality    25,535,832
  meteorology     8,838,144

Time coverage: 2001-01-01 00:00:00 -> 2024-12-31 23:00:00
```

**Shared window for selected pollutants [NO2, O3, <PM10, SO2]:** 2001-01-01 -> 2024-12-31. All four pollutants span the full period, so no window restriction is needed for cross-pollutant analyses.

**Stable-sensor check for NO2:** 13 of 24 sensors are present in ≥80% of the months covered by the most active sensor. This filter is applied throughout Task 4 to prevent composition effects as the sensor network expands over time.

### Descriptive Statistics

**Air quality variables:**

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Variable</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Count</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Mean</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Std</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Median</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Max</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">NO2</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">4,128,864</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">39.06 µg/m³</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">30.42</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">31</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">586</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">O3</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">2,323,032</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">47.53 µg/m³</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">32.91</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">46</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">235</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;"><PM10</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">2,077,584</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">22.11 µg/m³</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">20.34</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">17</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">721</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">SO2</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">1,779,480</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">8.29 µg/m³</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">7.32</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">7</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">199</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">NO</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">4,128,864</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">23.71 µg/m³</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">52.17</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">6</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">1502</td>
  </tr>
  </tbody>
</table>
**Meteorology - notable anomalies:**

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Variable</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Min recorded</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Expected range</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Assessment</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">TEMP</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">−318 °C</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">−10 to 45 °C</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Sensor hardware fault</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">HR</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">−149%</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">0-100%</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Sensor fault</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">PRE</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">0 hPa</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">880-1050 hPa</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">414 records - impossible vacuum</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">VV</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">0 m/s (max 989 m/s)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">0-30 m/s</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Extreme outlier</td>
  </tr>
  </tbody>
</table>
These anomalies motivate the physical validity checks in Task 2.

---

### Figure: task1_no2_monthly.png

![NO2 Monthly Trend and Sensor Count](figures/task1_no2_monthly.png)

**What this shows:** Two panels. Top: monthly mean NO2 (µg/m³) 2001-2024 comparing all sensors (blue) vs. stable-sensor subset (orange). Bottom: count of active NO2 sensors by month.

**Why we made it:** Verifies that apparent trends are not caused by network expansion. If new sensors join in areas with lower pollution, the city-wide mean would artificially decline even if individual stations showed no change. The near-identical blue/orange lines confirm this composition bias is minimal.

**Key observations:** A clear long-run decline from ~55 µg/m³ (2001) to ~30 µg/m³ (2020-2024). COVID-19 lockdown dip visible in 2020. Bottom panel shows sensor count growing from ~10 (2001) to 24 (by 2010), confirming why the stable-sensor filter is necessary.

---

### Figure: task1_air_quality_distributions.png

![Air Quality Variable Distributions](figures/task1_air_quality_distributions.png)

**What this shows:** Histograms for all 14 air quality variables (40 bins each, max 100,000 points sampled for rendering speed).

**Why we made it:** Distribution inspection reveals the empirical range, identifies sensor faults at extreme values, and informs downstream model choices.

**Key observations:**
- NO, NO2, NOX: Strongly right-skewed. Rush-hour spikes create long tails; most hours are low.
- O3: Bimodal - near zero at night (consumed by NO via titration), moderate-high during daylight photochemical production.
- CO: Degenerate, near-zero - modern Madrid vehicles produce almost no measurable CO.
- HIDROCARBS_TOTALES: Tight cluster at ~1 ppm - constant background.
- BENCENO, TOLUENO, ETILBENCENO: Near-zero-centered, consistent with benzene-family VOCs at trace concentrations.

---

### Figure: task1_meteorology_distributions.png

![Meteorology Distributions](figures/task1_meteorology_distributions.png)

**What this shows:** Histograms for 7 meteorological variables.

**Why we made it:** Meteorological variables are predictors in Tasks 8 and 9. Understanding their distributions reveals which need transformation and which contain artifacts.

**Key observations:**
- TEMP: Approximately normal, centred at ~15°C - Madrid's temperate continental climate.
- HR: Bimodal - dry summer (~40%) vs. humid winter/spring (~75%).
- VV: Heavy right skew. Most hours calm; rare windy episodes. 989 m/s max = sensor fault.
- DV: Roughly uniform 0-360° with mild SW preference (Atlantic flow over Iberian plateau).
- PRE: Normal around 936 hPa (Madrid at 667m elevation). 414 records at 0 hPa are physically impossible - flagged in Task 2.
- RS: Bimodal: zero at night, high at midday. Slight negative values are calibration artefacts.
- PRECIPITACION: Zero-inflated (Madrid is semi-arid); long tail from storm events.

---

### Figure: task1_traffic_distributions.png

![Traffic Variable Distributions](figures/task1_traffic_distributions.png)

**What this shows:** Histograms for all 15 traffic variables (3 categories × 5 interpolation methods).

**Why we made it:** Immediately reveals which interpolation methods produce physically meaningful values and which are numerically unstable - critical before using traffic as predictors.

**Key observations:**
- **RBF-Gaussian (TI, SP, OC):** Distributions collapse entirely to near-zero. Gaussian basis functions decay as exp(−r²/ε²) - for a sparse city-wide sensor network, they predict near-zero everywhere away from support points. Physically useless.
- **RBF-Multiquadric (TI):** Extreme outliers up to ~2.8×10¹⁸ vehicles/hour - clear numerical overflow. Excluded from predictive models.
- **IDW and Kriging:** Produce physically reasonable distributions (250-400 veh/h for TI, 15-25 km/h for SP, 5-8% for OC). **These are the traffic variables used in Tasks 8 and 9.**
- **RBF-Linear:** Also reasonable, as the linear basis function avoids oscillation artifacts.

---

## 6. Task 2 - Missingness & Data Quality

### Goal

Quantify original missingness using the `is_interpolated` flag, characterize its temporal and spatial structure, and detect physically invalid or inconsistent values.

### Why `is_interpolated` and Not NaN

METRAQ has filled every missing value before release - there are **zero NaN values** in the raw data. The `is_interpolated == True` flag marks positions that were originally absent and filled by the dataset authors. We reverse-engineer original missingness from this flag, stripping `is_interpolated == True` entries before applying our own imputation. This is the correct methodology: we treat METRAQ's interpolation as the "official" fill that we will compare against.

### Why Distinguish Temporal vs. Sensor-Specific Gaps

The project brief explicitly requires this distinction because the two types have different causes and require different imputation strategies:
- **Sensor-specific gaps** (one station missing while others report): ideal for **spatial KNN**, which uses neighboring sensors at the same timestamp.
- **Temporal gaps** (long consecutive outages): KNN fails here; temporal methods (rolling mean) or cross-pollutant approaches are needed.

### Missingness by Variable

```
         magnitude_name   missing_rate
                    PRE       87.68%  <- meteorology only from 2019
                     RS       87.61%
                     DV       83.63%
                     VV       83.58%
          PRECIPITACION       79.29%
                     HR       54.64%
                   TEMP       51.07%
            ETILBENCENO        5.39%
                <PM2.5         3.43%
                  <PM10        2.86%
                    NO2        1.76%
             TI_KRIGING        0.00%  <- traffic: always zero by construction
```

Meteorological variables show 50-88% missing because they were introduced in 2019: (17 pre-2019 years / 24 total years) ≈ 71% of the time series is necessarily interpolated from sparse post-2019 observations. All traffic variables have 0% missingness - they are the output of interpolation algorithms by construction, so there is no concept of "originally missing" for them.

### Consecutive Gap Analysis (Top 10 by Max Gap)

```
     sensor_name   magnitude  max_gap_h  n_gaps  total_missing_h
      Villaverde         SO2     31,226     340           38,229
      Villaverde          NO     31,226     416           38,457
      Villaverde         NOX     31,226     417           38,458
      Villaverde         NO2     31,226     417           38,458
      Villaverde          O3     31,226   1,053           39,324
Escuelas Aguirre   HIDROCARBS    19,697     450           21,876
Escuelas Aguirre      METANO     19,697     445           21,796
Escuelas Aguirre     TOLUENO     16,148   1,161           22,578
Escuelas Aguirre      BENCENO    16,148     988           22,055
```

**31,226 consecutive hours ≈ 3.57 years** at Villaverde (all major pollutants simultaneously absent). This station was completely offline from approximately 2004 to 2008. This gap is too long for any 24-hour temporal imputation method - only spatial KNN (borrowing from neighbor stations) can bridge it. This finding directly informed the method selection in Task 3.

### Physical Validity Checks

```
Physically impossible zeros (PRE = 0 hPa):     414  <- atmospheric pressure cannot be zero
Negative values where impossible:              684
  -> HR: 504 records (humidity cannot be negative)
  -> RS:  180 records (solar radiation cannot be negative in non-calibration hours)

Bounded-range violations:
  Invalid HR (outside 0-100%): 771
  Invalid DV (outside 0-360°):   0  <- wind direction is clean
  Invalid PREC (< 0 mm):         0  <- precipitation is clean

Duplicate rows on (sensor_id, magnitude_id, entry_date): 0
Sensors with inconsistent coordinates:                    0
```

**IQR outliers (k=3 per sensor), top 5:**

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Variable</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">n_outliers</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Cause</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">NO</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">338,123</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Real pollution spikes (combustion)</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">NOX</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">162,031</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Real spikes (NO + NO2)</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">PRE</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">73,339</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Sensor malfunctions</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">OC_RBF_MULTIQUADRIQ</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">72,812</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Numerical interpolation artefacts</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">OC_RBF_GAUSSIAN</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">57,145</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Numerical artefacts</td>
  </tr>
  </tbody>
</table>
NO and NOX outliers are mostly **real pollution episodes** - short-duration high-concentration rush-hour events - not sensor faults. The IQR method flags them because the distribution is heavily right-skewed, not because the values are erroneous.

---

### Figure: task2_missingness_sensor_var.png

![Missingness by Sensor × Variable](figures/task2_missingness_sensor_var.png)

**What this shows:** Heatmap where rows = 24 monitoring stations, columns = 36 variables, cell colour = fraction of `is_interpolated == True` for that (sensor, variable) pair.

**Why we made it:** A single-row summary table shows overall missingness, but this heatmap reveals which specific stations are worst affected and whether missingness is uniform across the network. This is essential for deciding which sensors to include in each analysis.

**Key observations:**
- Right side (meteorological variables): Near-complete missingness (dark purple) at almost all stations - a thick band reflecting 17 years of pre-2019 interpolation.
- Traffic columns (left/center): Uniform zero missingness (bright yellow) everywhere.
- Villaverde row: Noticeably darker in air quality columns, confirming the multi-year outages.
- Escuelas Aguirre: Second-darkest in air quality columns, consistent with the 16,000-19,000 hour gaps.

---

### Figure: task2_missingness_temporal.png

![Temporal Missingness Heatmap](figures/task2_missingness_temporal.png)

**What this shows:** Heatmap where x-axis = calendar month (2001-2024), y-axis = variable, cell colour = average missingness rate for that month.

**Why we made it:** Reveals whether missingness is uniformly distributed over time or concentrated in specific periods - crucial for choosing analysis windows.

**Key observations:**
- Meteorological variables (top rows): Sharp transition at 2019. Before 2019: solid dark (100% missing). After 2019: lighter shades reflecting the sensor operational period.
- Air quality variables (middle rows): Scattered patches throughout all 24 years - maintenance and equipment failures with no seasonal pattern.
- Traffic variables: Uniformly zero from 2015 onward.

---

## 7. Task 3 - Imputation

### Goal

Replace originally-missing values with estimates that do not leak future information, compare methods quantitatively against each other and against METRAQ's baseline, and select the best causal method per pollutant for all downstream analyses.

### The Causal Discipline Principle

The most important design principle of this task: **no imputation method may use observations from future timestamps**. In a real deployment, imputation would be applied in real-time - looking ahead at what NO2 will be tomorrow to fill today's gap is operationally impossible and scientifically inappropriate. Therefore:
- **ffill/bfill 24h** uses backward-fill -> looks into the future -> **excluded from final selection** (kept only as a theoretical ceiling benchmark).
- **KNN spatial IDW** and **rolling_24h_past** are the only admissible methods for the final imputed dataset.

### Why Three Methods Were Implemented

**1. KNN spatial IDW (k = 3 nearest sensors)**
- **Physical motivation:** Air quality is spatially correlated - nearby sensors experience similar conditions simultaneously. A weighted average of neighbors at the exact same timestamp is the most direct estimate of what a missing sensor would have measured.
- **Why k = 3:** k=1 is too sensitive to a single noisy/malfunctioning neighbor; k=5 begins averaging over sensors in different urban districts with different emission profiles. k=3 provides robust averaging while remaining locally representative.
- **How it works:** For each missing (timestamp, sensor) position, compute the inverse-distance-weighted mean of the k=3 nearest sensors that have observed values at that exact hour. If all k neighbors are also missing, fall back to an expanding mean of all past values at the same sensor (causally safe).
- **Why it matches METRAQ's approach:** METRAQ uses spatial interpolation (Kriging, IDW, etc.) as its primary method. Our KNN is a simpler version of the same idea, making it the most natural comparison.

**2. Rolling 24h past mean**
- **Physical motivation:** Pollution is highly autocorrelated at hourly timescales. The past 24 hours at the same sensor encapsulate one full diurnal cycle of traffic and photochemical patterns.
- **Why 24 hours:** Captures the full daily periodicity (morning rush, midday photolysis, evening rush, nighttime stability) - the dominant short-term pattern in all four pollutants.
- **When it wins over KNN:** When SO2 emission sources are highly localized (specific industrial facilities, specific bus routes), different sensors have very different SO2 profiles even if nearby. Past values from the same sensor are more informative than neighbor values.

**3. ffill/bfill 24h (non-causal baseline)**
- **Why kept:** Quantifies the "information value of the future" - how much better we could do if we had access to future observations. Its lower RMSE tells us the upper bound on causal methods.
- **Why excluded from final selection:** `bfill()` uses the next available observation, which is a future value. Operationally inadmissible.

### Pseudo-Gap Evaluation Results

3% of observed values are randomly masked per sensor, each method is applied, and RMSE/MAE are computed against known true values:

```
magnitude_name    method                rmse      mae
         <PM10    knn                  12.075    7.500  <- best causal
         <PM10    ffill_bfill_24h      12.226    6.928  (non-causal baseline)
         <PM10    rolling_24h_past     16.190   10.167
           NO2    ffill_bfill_24h      13.790    8.841  (non-causal)
           NO2    knn                  16.585   11.306  <- best causal
           NO2    rolling_24h_past     24.292   17.583
            O3    ffill_bfill_24h      11.109    7.424  (non-causal)
            O3    knn                  12.664    8.990  <- best causal
            O3    rolling_24h_past     24.196   18.597
           SO2    ffill_bfill_24h       2.573    0.999  (non-causal)
           SO2    rolling_24h_past      4.236    1.999  <- best causal
           SO2    knn                   5.521    3.343
```

**Why KNN wins for NO2, O3, PM10:**
These are city-wide atmospheric phenomena. Photochemical production (O3), traffic exhaust (NO2), and dust episodes (PM10) affect multiple stations simultaneously. Spatial neighbors at the same timestamp carry the most relevant concurrent information.

**Why rolling_24h_past wins for SO2:**
SO2 has more localized emission sources (specific industrial plants, specific bus routes using higher-sulfur fuel). Different sensors have distinct SO2 profiles even when geographically close. The sensor's own past captures its specific local regime better than an average of neighbors.

### Fill Coverage - A Key Metric Beyond Accuracy

```
Pollutant   Method             n_missing    n_filled  pct_filled
NO2         METRAQ              107,881       72,817      67.5%
NO2         ffill_bfill         107,881       29,264      27.1%
NO2         knn                 107,881      107,880     100.0%  <- always fills
NO2         rolling             107,881      102,132      94.7%
O3          METRAQ               88,945       53,881      60.6%
O3          knn                  88,945       88,945     100.0%
SO2         METRAQ               80,972       45,908      56.7%
SO2         knn                  80,972       80,972     100.0%
```

**Why ffill/bfill fills only 18-34%:** It requires an observed value within 24 hours on either side. For Villaverde's 3.57-year outage, no observation exists within 24 hours -> method fails entirely for that gap.

**Why METRAQ itself fills only 57-77%:** The dataset authors' sophisticated spatial interpolation also cannot bridge extreme multi-year gaps - the remaining 23-43% are left as NaN in the released dataset.

**Why KNN achieves 100%:** Spatial IDW always finds at least one neighbor with a value (24 sensors, rarely all missing simultaneously), and the expanding-mean fallback handles the startup case.

### KS-Test: Distribution Similarity to METRAQ

```
Pollutant   Method              ks_stat   (0 = identical to METRAQ)
NO2         knn                  0.0679  <- most similar to METRAQ
NO2         rolling_24h_past     0.3578  <- most different
O3          knn                  0.1937
O3          rolling_24h_past     0.2694
<PM10       knn                  0.0706  <- most similar
<PM10       rolling_24h_past     0.3336
SO2         rolling_24h_past     0.3697  <- wins on RMSE but very different distribution
```

Rolling_24h_past consistently shows the highest KS distance: it smooths out the high-frequency variation, producing narrower distributions (lower variance, higher median) than the true missing-value distributions. KNN better preserves the shape because it borrows from neighbors experiencing the same concurrent conditions.

### Final Method Selection

```
Selected final causal imputation method per pollutant (lowest RMSE):
  NO2   -> knn
  O3    -> knn
  <PM10 -> knn
  SO2   -> rolling_24h_past
```

### Residual NaN Check After Final Imputation

```
<PM10: 1 NaN of 2,095,128 (0.000%)
NO2:   1 NaN of 4,163,928 (0.000%)
O3:    0 NaN of 2,358,096 (0.000%)
SO2:   0 NaN of 1,814,544 (0.000%)
Total: 2 residual NaNs of 10,431,696 rows (0.000%)
```

Two NaNs at timestamp t=0 where the expanding-mean fallback has no prior observations. Left as NaN intentionally - forcing a fill with zero or a constant would misrepresent coverage.

---

### Figure: task3_distribution_NO2.png

![NO2 Imputation Distribution](figures/task3_distribution_NO2.png)

**What this shows:** KDE density curves for imputed values at originally-missing positions for NO2. All four methods compared.

**Why KDE:** Smooth density estimate allows visual comparison of distribution shape (peak location, spread, tail behavior) that histograms with fixed bins obscure.

**Key observations:** METRAQ (dark grey) peaks at ~35-45 µg/m³. KNN (green) closely tracks METRAQ's shape - similar peak location and right tail. Rolling (red) is notably shifted right (~55 µg/m³ peak) and much narrower - the 24h averaging smooths out high/low extremes, producing overly central values. ffill/bfill (blue) tracks METRAQ but is narrower because it only fills the 27% "easy" gaps (near observed values), missing the harder long-gap cases.

---

### Figure: task3_distribution_O3.png

![O3 Imputation Distribution](figures/task3_distribution_O3.png)

**Key observations:** O3 shows two modes (nighttime near-zero and daytime photochemical values). KNN captures both modes. Rolling eliminates the near-zero mode because it averages over day and night, always producing a value between 20-60 µg/m³.

---

### Figure: task3_distribution_PM10.png

![PM10 Imputation Distribution](figures/task3_distribution_PM10.png)

**Key observations:** PM10 shows the best KNN-METRAQ agreement (KS=0.071). PM10 episodes (dust, inversions) are truly city-wide - all sensors see the same event simultaneously, making spatial averaging ideal.

---

### Figure: task3_distribution_SO2.png

![SO2 Imputation Distribution](figures/task3_distribution_SO2.png)

**Key observations:** For SO2, rolling_24h_past wins on RMSE but its distribution (narrow peak) is very different from METRAQ's broader distribution. This highlights a limitation: minimizing RMSE at pseudo-gap test points does not guarantee the imputed distribution is realistic, especially for localized pollutants.

---

### Figure: task3_rmse_comparison.png

![Imputation RMSE Comparison](figures/task3_rmse_comparison.png)

**What this shows:** Grouped bar chart of pseudo-gap RMSE by pollutant and causal method (KNN vs. rolling_24h_past).

**Key observations:** KNN is clearly better for NO2, O3, and PM10 (shorter bars). Rolling wins for SO2. The largest RMSE gap is for O3 (KNN 12.7 vs rolling 24.2) - O3's exceptionally strong spatial coherence makes KNN nearly ideal, while rolling fails because the 24h average smooths out the strong day/night cycle that is O3's most prominent feature.

---

## 8. Task 4 - Temporal Analysis

### Goal

Characterize seasonal cycles, long-run emission trends, and diurnal traffic-driven patterns. Quantify statistical significance of seasonal variation. Study whether patterns are consistent across individual stations or geographically heterogeneous.

### Why Three Time Granularities

The project brief asks to "choose time granularity and justify it":

1. **Yearly** - captures long-run policy-driven trends (Euro emission standards tightening every ~5 years, COVID-19 lockdown, fuel transitions).
2. **Monthly** - captures the dominant seasonal cycle: winter heating/stability vs. summer photochemistry operate at the monthly timescale.
3. **Hourly (diurnal)** - captures traffic-driven daily patterns directly relevant to interventions like low-emission zones and time-of-day traffic bans.

A single granularity would miss the others. Yearly data hides the 2× seasonal variation; hourly data cannot reveal the 20-year declining trend.

### Why Station-Normalized Aggregation

`station_normalized_group_mean()` performs two-step aggregation: (1) mean within each sensor for the time period, (2) mean across sensors. This ensures sensors with denser measurement schedules do not dominate the city-wide average. Without this, a station measuring 10 variables would contribute 10× more rows to the average than a station measuring 1 variable.

### Why the Stable-Sensor Filter (≥80% of Max Months)

`build_stable_sensor_subset()` retains only sensors present in ≥80% of the months covered by the most active sensor. Without this filter:
- A sensor that starts in 2015 in a lower-pollution suburban area creates a spurious apparent decline from 2015 onwards.
- A station that goes offline in 2010 creates a spurious apparent increase when its below-average readings disappear from the average.

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Pollutant</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Total sensors</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Stable sensors</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">NO2</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">24</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">13</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">O3</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">14</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">7</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;"><PM10</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">13</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">6</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">SO2</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">10</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">9</td>
  </tr>
  </tbody>
</table>
### Statistical Test: Kruskal-Wallis (Seasonality)

The H-test checks whether 12 monthly groups come from the same distribution. Used instead of ANOVA because pollution distributions are not normal. **Only real (non-interpolated) observations** are used to avoid inflated significance from smoothed imputed values.

```
Pollutant   H-statistic   p-value   Interpretation
NO2         214,513.2      ~0       Highly significant seasonal variation
O3          430,164.2      ~0       Even stronger (photochemical cycle)
<PM10        42,936.8      ~0       Significant
SO2         124,320.9      ~0       Significant
```

All four pollutants show statistically significant seasonal variation (p ≈ 0).

---

### Figure: task4_real_vs_imputed_all.png

![Real vs Imputed Monthly Mean](figures/task4_real_vs_imputed_all.png)

**What this shows:** Four panels (one per pollutant), monthly mean 2001-2024 comparing "real measurements only" (solid blue) vs. "after imputation" (dashed orange).

**Why we made it:** Directly validates whether imputation introduces systematic bias into temporal analysis. If imputed values were biased, the orange line would drift systematically from blue.

**Key observations:** The two lines track each other extremely closely for all four pollutants throughout the entire 24-year period. Small divergences appear in early years (thinner coverage), but no systematic bias is visible. This validates the imputation methodology and justifies using the imputed series for all subsequent temporal analyses.

---

### Figure: task4_yearly_NO2.png

![NO2 Yearly Trend](figures/task4_yearly_NO2.png)

**What this shows:** Annual city-wide NO2 average (stable sensors, station-normalized) 2001-2024.

**Key observations:** Strong declining trend from ~54 µg/m³ (2001) to ~30 µg/m³ (2020-2024) - a ~44% reduction. This reflects: progressive Euro emission standard tightening (Euro 3->6d), electrification of Madrid's bus/taxi fleet, introduction of the Madrid Central Low Emission Zone (2018). COVID-19 lockdown dip in 2020, partial recovery in 2021-2022. The 2024 value is close to the EU annual limit of 40 µg/m³, with some sensors likely exceeding it.

---

### Figure: task4_yearly_O3.png

![O3 Yearly Trend](figures/task4_yearly_O3.png)

**Key observations:** A slight *upward* trend in O3, roughly inverse to NO2's decline. This is the **ozone penalty**: O3 is partly suppressed by NO via the titration reaction O3 + NO -> NO2 + O2. As urban NO decreases (cleaner vehicles), less O3 is consumed - background ozone accumulates. This well-documented phenomenon creates a trade-off between NO2 and O3 management in European cities.

---

### Figure: task4_yearly_PM10.png

![PM10 Yearly Trend](figures/task4_yearly_PM10.png)

---

### Figure: task4_yearly_SO2.png

![SO2 Yearly Trend](figures/task4_yearly_SO2.png)

Both PM10 and SO2 decline substantially. SO2 shows the steepest drop (~12 µg/m³ -> ~4 µg/m³), reflecting EU sulfur cap regulations that mandated low-sulfur fuel and phased out sulfurous heating oil.

---

### Figure: task4_monthly_NO2.png

![NO2 Monthly Seasonality](figures/task4_monthly_NO2.png)

**What this shows:** Average NO2 by calendar month.

**Key observations:** Winter maximum (December-January, ~50 µg/m³), summer minimum (July-August, ~25 µg/m³). Driven by: (1) reduced solar radiation in winter -> slower photolytic NO2 destruction; (2) temperature inversions trapping pollutants; (3) cold-start engine emissions.

---

### Figure: task4_monthly_O3.png

![O3 Monthly Seasonality](figures/task4_monthly_O3.png)

**Key observations:** *Opposite* to NO2 - minimum in winter (~20 µg/m³), maximum in June-July (~75 µg/m³). More sunlight -> more photolysis -> more O3 formation. Lower NO in summer also means less O3 titration.

---

### Figure: task4_monthly_PM10.png

![PM10 Monthly Seasonality](figures/task4_monthly_PM10.png)

---

### Figure: task4_monthly_SO2.png

![SO2 Monthly Seasonality](figures/task4_monthly_SO2.png)

PM10 shows mild winter elevation plus a spring peak from Saharan dust transport events. SO2 winter maximum reflects heating-season combustion.

---

### Figure: task4_hourly_NO2.png

![NO2 Diurnal Profile](figures/task4_hourly_NO2.png)

**What this shows:** Average NO2 by hour of day (0-23), with morning rush (07-10h) and evening rush (17-20h) shaded.

**Key observations:** Two clear rush-hour peaks at 08-09h and 19-20h. Midday trough from photolytic destruction (UV at maximum). Nighttime intermediate values reflect atmospheric stability trapping residual emissions.

---

### Figure: task4_hourly_O3.png

![O3 Diurnal Profile](figures/task4_hourly_O3.png)

**Key observations:** Near-zero at night (O3 consumed by NO), rising from ~08h as solar radiation begins, peaking ~13-15h, declining in the evening. The classic photochemical daytime cycle.

---

### Figure: task4_hourly_PM10.png

![PM10 Diurnal Profile](figures/task4_hourly_PM10.png)

---

### Figure: task4_hourly_SO2.png

![SO2 Diurnal Profile](figures/task4_hourly_SO2.png)

PM10 shows a diffuse morning rush pattern. SO2 is relatively flat - consistent with heating and power generation as primary sources (less diurnal than vehicle traffic).

---

### Figure: task4_heatmap_NO2.png

![NO2 Year × Month Heatmap](figures/task4_heatmap_NO2.png)

**What this shows:** Heatmap with year (2001-2024) on y-axis, month (1-12) on x-axis, colour = average NO2.

**Why this is powerful:** Simultaneously shows the seasonal cycle (column pattern) and the long-term trend (row-wise gradient) without losing information from either dimension.

**Key observations:** Alternating high-winter / low-summer bands in every row (seasonal cycle). Overall intensity decreasing from top to bottom (declining trend). 2020 row noticeably lighter than neighbors (COVID lockdown).

---

### Figure: task4_heatmap_O3.png

![O3 Year × Month Heatmap](figures/task4_heatmap_O3.png)

---

### Figure: task4_heatmap_PM10.png

![PM10 Year × Month Heatmap](figures/task4_heatmap_PM10.png)

---

### Figure: task4_heatmap_SO2.png

![SO2 Year × Month Heatmap](figures/task4_heatmap_SO2.png)

O3 shows the complementary pattern (summer peaks, slight upward trend). PM10 and SO2 both show declining trends with winter peaks.

---

### Figure: task4_sensor_trends_NO2.png

![NO2 Per-Sensor Yearly Trends](figures/task4_sensor_trends_NO2.png)

**What this shows:** Individual yearly trends for the top-6 stable NO2 sensors.

**Key observations:** All stations decline, but high-traffic stations (Castellana, Cuatro Caminos) have higher absolute values and steeper reductions from emission controls. Green/suburban stations (El Pardo - royal forest) are consistently lower and change less. This confirms traffic emission controls as the primary driver.

---

### Figure: task4_sensor_trends_O3.png

![O3 Per-Sensor Yearly Trends](figures/task4_sensor_trends_O3.png)

---

### Figure: task4_sensor_trends_PM10.png

![PM10 Per-Sensor Yearly Trends](figures/task4_sensor_trends_PM10.png)

---

### Figure: task4_sensor_trends_SO2.png

![SO2 Per-Sensor Yearly Trends](figures/task4_sensor_trends_SO2.png)

O3 sensor trends are tightly clustered (city-wide photochemical uniformity). PM10 diverges more (local dust sources). SO2 was highly variable early (multiple industrial sources) and converged to a narrow low band after fuel regulation.

---

## 9. Task 5 - Spatial Network

### Goal

Build a graph where nodes = monitoring stations and edges = geographic proximity. Study structural properties (degree distribution, connected components, community structure) under different connectivity assumptions, and explain why naive approaches fail.

### Why the Naive Fully-Connected Graph Is Useless

```
Naive fully-connected graph:
  Nodes: 24  |  Edges: 276  |  Density: 1.0000
  Implication: density=1 -> no hub/periphery, community detection meaningless.
```

With all 276 possible edges present, the network has no structure. Every sensor is equally "neighbors" with every other - there is no hub, no peripheral node, no community. Community detection algorithms assign all 24 nodes to one community. Average shortest path = 1 (trivially). This graph is useless for any structural analysis.

### Why KNN Instead of Distance Threshold

The threshold-based approach produces disconnected graphs at low thresholds:

```
graph_name     n_nodes  n_edges  density  connected  components  avg_degree
threshold_3km       24       26    0.094      False          11        2.17
threshold_5km       24       73    0.265      False           2        6.08
threshold_7km       24      127    0.460       True           1       10.58
knn_k2              24       32    0.116       True           1        2.67  <- selected
knn_k3              24       50    0.181       True           1        4.17
knn_k4              24       68    0.246       True           1        5.67
```

At 3 km: 11 disconnected components - KNN is the only approach that guarantees connectivity for any k.

### Why KNN k = 2 (Smallest Connected k)

**KNN k = 2 is the minimum k that yields a connected graph.** Connectivity is required because:

1. **Average shortest path length** is undefined on disconnected graphs.
2. **Modularity-based community detection** produces trivial singleton communities for isolated nodes.
3. **Physical justification**: In a single metropolitan area, it is reasonable to assume every sensor can be "reached" through the network - a station completely isolated from all others is a modeling artifact, not a real spatial relationship.

Choosing the *smallest* connected k preserves maximum structural information. Larger k adds long-range edges that blur local neighborhood structure without adding physical meaning.

### Terminal Output - Final Graph

```
Final spatial graph (KNN k=2):
  Nodes: 24  |  Edges: 32  |  Density: 0.116
  Connected: True  |  Components: 1
  Avg degree: 2.67  |  Avg clustering: 0.368
  Avg shortest path: 4.82
  Communities: 5  |  Modularity: 0.613
```

Modularity 0.613 confirms strong community structure - sensors cluster into geographic zones with denser internal connections.

### Community Assignments

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Community</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Stations</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Geographic zone</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">0</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Escuelas Aguirre, Parque del Retiro, Vallecas, Moratalaz, Méndez Álvaro, Ensanche de Vallecas, Plaza del Carmen</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">East / centre-east</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">1</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Plaza de España, Ramón y Cajal, Arturo Soria, Cuatro Caminos, Castellana</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">North-centre / Castellana axis</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">2</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Tres Olivos, Barrio del Pilar, Sanchinarro, El Pardo, Plaza Castilla</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">North</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">3</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Villaverde, Plaza Elíptica, Casa de Campo, Farolillo</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">South-west</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">4</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Urb. Embajada, Juan Carlos I, Barajas Pueblo</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">North-east / Airport</td>
  </tr>
  </tbody>
</table>
---

### Figure: task5_degree_distribution.png

![Spatial Network Degree Distribution](figures/task5_degree_distribution.png)

**What this shows:** Histogram of node degrees in the KNN k=2 graph.

**Why we made it:** Degree distribution characterizes network topology. The project brief asks to "study and analyse the graph" - degree distribution is the first standard graph metric.

**Key observations:** Most nodes have degree 2 (exactly their two nearest neighbors). A few reach degree 3-5 because some sensors appear in multiple other sensors' KNN lists (incoming edges add to degree). Compact distribution with no high-degree hubs - consistent with a geographically uniform sensor deployment.

---

### Figure: task5_spatial_network.png

![Spatial Network with Communities](figures/task5_spatial_network.png)

**What this shows:** NetworkX visualization of the 24-node graph with sensors at actual UTM coordinates (in km). Edges shown in grey, nodes coloured by community.

**Why we made it:** Primary network visualization. Positioning nodes at actual geographic coordinates lets the reader directly interpret community membership in terms of urban geography without any prior knowledge of Madrid.

**Key observations:** Five color groups cluster clearly by geography. Community 4 (northeast) near Barajas airport. Community 3 (southwest) in the Villaverde industrial zone. Community 1 runs along the north-south Castellana axis. The sparse edge structure (2-3 edges per node) makes neighborhood relationships clear without visual clutter.

---

### Figure: task5_sensor_locations_communities.png

![Sensor Locations with Communities](figures/task5_sensor_locations_communities.png)

**What this shows:** Cleaner scatter plot of sensor UTM coordinates with community colour and text labels.

**Key observations:** Geographic correspondence with Madrid's actual district structure is striking. Community 2 includes El Pardo (royal forest, northwest) - a distinctly green zone. Community 3 includes Villaverde (main industrial zone, south). Community 1 follows the Paseo de la Castellana corridor.

---

## 10. Task 6 - Correlation Network

### Goal

Build a behavioral association network connecting sensors that co-vary over time, compare it against the spatial network, and analyze how the network structure changes with different thresholds and time windows.

### Why Behavioral Correlation Instead of Just Proximity

The project brief poses the key question: *"Are two areas necessarily related just because they are nearby?"* The answer is no. Two sensors 500 m apart - one in a park, one on a motorway - may be entirely uncorrelated behaviorally. Two sensors 15 km apart may be highly correlated if they both respond to city-wide events (Saharan dust, temperature inversions). **Behavioral correlation captures shared drivers (meteorology, traffic rhythms, heating season) that proximity-based networks cannot.**

### Why Monthly Aggregation for Correlations

1. **Signal/noise:** Hourly Pearson correlations are noisy. Monthly aggregation smooths measurement noise and short-duration events.
2. **Missing-data robustness:** min_periods=12 requires ≥ 1 full year of overlapping monthly values before reporting a correlation - prevents spurious high correlations from a few months.
3. **Seasonal structure:** Monthly means capture the dominant seasonal signals that drive long-range spatial correlations.

### Why Signed Edges (Both Positive and Negative)

Most correlation networks only include positive edges. We include both positive (r > 0, |r| ≥ 0.6) and negative (r < 0, |r| ≥ 0.6) edges because:
- **Positive edge:** both sensors rise and fall together - shared emission source or shared meteorological driver.
- **Negative edge within a single pollutant:** sensors are anti-correlated. Can reflect upwind/downwind timing differences, or opposing local microenvironments.
- The `weight` attribute stores |r| for modularity calculations (non-negative); the `corr` attribute preserves the signed r for physical interpretation.

### Terminal Output - NO2 Network (Fully Connected)

```
[NO2] Threshold sweep:
  |r| >= 0.50: 276 edges, density=1.000
  |r| >= 0.60: 276 edges, density=1.000  <- fully connected at our threshold
  |r| >= 0.70: 272 edges, density=0.986
  |r| >= 0.80: 245 edges, density=0.888

[NO2] Final graph (thr=0.6): 24 nodes, 276 edges, modularity=0.0000
Max |r_raw - r_detrended| across all pairs: 0.266
WARNING: some correlations partly reflect shared downward trend.
Raw-only sensitivity: Jaccard = 1.000 (identical to imputed graph)
```

At threshold 0.60, **all 276 possible NO2 sensor pairs have |r| ≥ 0.60** - a fully connected graph with zero modularity (no community structure). This means all 24 Madrid NO2 sensors are strongly behaviorally synchronized. Physical explanation: NO2 is driven by city-wide seasonal and diurnal cycles (heating season, traffic rhythms, photolysis) that affect all stations simultaneously. The detrending check shows that part of the raw Pearson r reflects the shared long-run declining trend - but even detrended correlations remain high (minimum ≈ 0.58 after first-differencing).

### Terminal Output - SO2 (Most Differentiated)

```
[SO2] Threshold sweep:
  |r| >= 0.60: 25 edges, density=0.556, components=2  <- not fully connected
  |r| >= 0.70: 19 edges, density=0.422, components=3
  |r| >= 0.80:  4 edges, density=0.089, components=7

Sanchinarro correlation range: r = 0.28-0.53 (outlier station)
```

SO2 shows real network structure at threshold 0.6: not all sensors are connected. Sanchinarro (northern suburban station) has consistently low correlations with all others, reflecting its distinct local SO2 profile far from central industrial and heating corridors.

### Terminal Output - Spatial vs. Correlation Comparison

```
Property               Spatial (KNN k=2)    Correlation NO2 (|r|≥0.6)
n_edges                        32                    276
density                    0.116                  1.000
avg_degree                  2.67                   23.0
avg_clustering              0.368                  1.000
```

The contrast is dramatic. The spatial network is sparse and structured (meaningful communities, meaningful degree distribution). The NO2 correlation network is a complete graph with no structure - every sensor belongs to the same behavioral community.

### Time-Window Comparison

```
Period          thr=0.6 edges   density at 0.7
2001-2012:           247           0.761
2013-2024:           276           0.996
```

Later period produces denser networks: (1) more sensors active simultaneously -> longer overlap -> lower correlation estimation noise; (2) improved city-wide air quality management may have genuinely increased spatial homogeneity of NO2.

---

### Figure: task6_corrmatrix_NO2.png

![NO2 Correlation Matrix](figures/task6_corrmatrix_NO2.png)

**What this shows:** 24×24 annotated Pearson correlation heatmap for NO2 (monthly means, min_periods=12). All cells are green because all r ≥ 0.68. The slight variation within the green range reveals which station pairs are slightly less synchronized - Plaza de España shows marginally lower correlations, consistent with its pedestrianized plaza microenvironment.

---

### Figure: task6_corrmatrix_NO2_real_only.png

![NO2 Raw-Only Correlation Matrix](figures/task6_corrmatrix_NO2_real_only.png)

**Why important:** Validates that high NO2 correlations are genuine data properties, not imputation artifacts. The matrix is visually indistinguishable from the imputed version (Jaccard = 1.000 at threshold 0.6).

---

### Figure: task6_corrmatrix_O3.png

![O3 Correlation Matrix](figures/task6_corrmatrix_O3.png)

Even higher correlations than NO2 (minimum r ≈ 0.88 after detrending). Ozone is the most spatially uniform pollutant - photochemical conditions (solar radiation, temperature) are essentially identical across a city-scale domain.

---

### Figure: task6_corrmatrix_PM10.png

![PM10 Correlation Matrix](figures/task6_corrmatrix_PM10.png)

First signs of heterogeneity: Plaza Elíptica appears lighter - its location near a major traffic interchange may produce a distinct PM10 source signature different from the general background dust.

---

### Figure: task6_corrmatrix_SO2.png

![SO2 Correlation Matrix](figures/task6_corrmatrix_SO2.png)

Clear differentiation. Sanchinarro row/column is visibly lighter than all others - consistently low r values (0.28-0.55) reflecting its distinct northern suburban SO2 profile.

---

### Figure: task6_corrnet_NO2.png

![NO2 Correlation Network](figures/task6_corrnet_NO2.png)

Fully connected complete graph for NO2 at threshold 0.6. Every sensor is directly connected to every other. The node geographic positioning remains informative - it shows that city-wide behavioral synchrony is independent of distance.

---

### Figure: task6_corrnet_NO2_real_only.png

![NO2 Raw-Only Network](figures/task6_corrnet_NO2_real_only.png)

Identical to the imputed NO2 network. Confirms robustness.

---

### Figure: task6_corrnet_O3.png

![O3 Correlation Network](figures/task6_corrnet_O3.png)

---

### Figure: task6_corrnet_PM10.png

![PM10 Correlation Network](figures/task6_corrnet_PM10.png)

---

### Figure: task6_corrnet_SO2.png

![SO2 Correlation Network](figures/task6_corrnet_SO2.png)

SO2 network clearly shows Sanchinarro as a peripheral node with fewer connections. The two disconnected components at threshold 0.6 are visible.

---

### Figure: task6_timewindow_comparison.png

![Time Window Comparison](figures/task6_timewindow_comparison.png)

**What this shows:** Two panels showing n_edges (left) and density (right) vs. Pearson threshold for the early (2001-2012) and late (2013-2024) NO2 periods.

**Key observations:** At all thresholds, the late period maintains more edges and higher density. The divergence is largest at intermediate thresholds (0.6-0.7). This suggests modern Madrid's NO2 distribution has become more spatially homogeneous - consistent with the hypothesis that city-wide emission controls equalize pollution levels across districts.

---

## 11. Task 7 - Propagation Modeling *(Optional)*

### Goal

Model how NO2 concentrations evolve and diffuse across the sensor network from one hour to the next using a physics-grounded spatial propagation model, then validate it on held-out test data.

### Why Graph Laplacian Diffusion - Detailed Physical Justification

The project brief asks to "design and implement a propagation model." We chose the **discrete-time Graph Laplacian Diffusion model** for the following reasons:

**1. Grounded in the physics of atmospheric dispersion:**
Pollutant transport in the atmosphere without wind is governed by Fick's diffusion law and its continuous-space equivalent, the heat equation: ∂C/∂t = D·∇²C. On a graph with N nodes, the spatial Laplacian ∇² is discretized to the graph Laplacian matrix **L = D_deg − A** (degree matrix minus adjacency). The random-walk Laplacian **L_rw = I − W** (where W is row-normalized inverse-distance weight matrix) gives the discrete heat equation:

```
X(t+1) = X(t) − α·L_rw·X(t)
        = (I − α·L_rw)·X(t)
        = ((1−α)·I + α·W)·X(t)
```

This is exactly the model implemented. The parameter α controls the diffusion speed.

**2. Mass conservation:**
Row normalization of W ensures Σ_j W[i,j] = 1 for all i. Therefore (1−α) + α·Σ_j W[i,j] = 1: the prediction for each sensor is always a convex combination of its own past value and neighbors' values. Total network concentration is preserved in each step - no pollutant is created or destroyed by the diffusion operator itself.

**3. Single interpretable parameter:**
α ∈ (0,1) has a clear physical meaning: what fraction of each sensor's next-hour value comes from spatial neighbors vs. from its own past. This can be tuned on a validation set and directly interpreted.

**4. Why NOT epidemic SIR/SIS models:**
SIR/SIS models are designed for discrete binary states (susceptible/infected/recovered) in population dynamics. They require thresholds (infection rate, recovery rate) and binary state transitions. NO2 concentration is a continuous quantity with no natural threshold for "infection." Forcing a continuous pollution field into a binary epidemic model would be physically unjustified and methodologically inappropriate.

**5. Why NOT pure machine learning:**
A Random Forest or neural network on spatial lag features would achieve higher R² but provide no physical insight. The Laplacian model makes a specific testable claim: *the diffusion rate is constant in time and space, and scales with inverse distance.* The unconstrained SAR Ridge benchmark shows what we lose and gain from this constraint.

### Model Setup

```
Weight matrix W: row-normalized inverse-distance from KNN k=2 graph
Non-zero entries: 64 of 576 (density = 11.1%)
Train hours: 168,307  |  Val hours: 16,831  |  Test hours: 42,077
α search grid: 20 values from 0.01 to 0.99
Best α = 0.113  (val RMSE = 10.0898)
```

### Model Comparison (Test Set)

```
                                  model     RMSE      MAE       R²
                   Persistence (1-step)  10.168    6.340    0.792
Laplacian Diffusion 1-step (α=0.113)    10.009    6.317    0.799  <- best physical
Laplacian Diffusion Recursive (α=0.113) 23.454   15.881   -0.106  <- error compounding
      Unconstrained SAR Ridge (1-step)   9.980    6.704    0.800

Diffusion-1step vs Persistence RMSE improvement: +1.56%
-> Neighbour information helps at the 1-hour horizon.
Recursive vs 1-step RMSE gap: +134.33% (error compounding under extrapolation)
```

**Physical interpretation of α = 0.113:**
Each hour, **11.3%** of the predicted NO2 at each sensor comes from spatial neighbors (inverse-distance weighted), and **88.7%** from its own previous value. The small α reflects a physical reality: at the 1-hour timescale, local emission sources (whether rush-hour traffic is happening at this specific intersection) dominate over wind transport from neighbors several km away. At 1 hour × ~1 m/s average wind speed = ~3.6 km transport - exactly at the margin of the KNN neighborhood.

**Why recursive diffusion fails (R² = −0.106):** With no emission forcing term, the model conserves mass but redistributes it until all sensors converge to the network mean. Within a few days of simulation, the predicted state is simply the spatiotemporal average - worse than the mean as a predictor. This is not a flaw in the model but an expected property of a pure diffusion operator: it cannot generate the episodic high concentrations driven by local emission events.

**SAR Ridge unconstrained coefficients:** β_self = +27.80, β_neighbor = +1.03, sum = +28.84. The unconstrained model is not physically interpretable (predictions would grow without bound over time) but fits slightly better at the 1-step horizon by acting as an amplified persistence model.

### Per-Sensor MAE

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Sensor</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">MAE (µg/m³)</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Location type</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">El Pardo</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">4.39</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Royal forest (northwest)</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Casa de Campo</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">4.46</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Large urban park</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Tres Olivos</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">5.27</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Northern suburban</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">...</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">...</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">...</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Ramón y Cajal</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">7.12</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Central arterial road</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Barajas Pueblo</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">7.83</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Airport access road</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Plaza Elíptica</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">7.95</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Major traffic interchange</td>
  </tr>
  </tbody>
</table>
Park/suburban sensors have lower errors (smooth, predictable NO2 from diffuse background sources). High-traffic peripheral sensors have higher errors (episodic spikes from specific local sources that the diffusion model cannot capture with a constant-rate linear operator).

---

### Figure: task7_alpha_tuning.png

![Alpha Tuning Curve](figures/task7_alpha_tuning.png)

**What this shows:** Validation RMSE vs. diffusion rate α (20 candidate values from 0.01 to 0.99). Red dashed line marks the optimal α = 0.113.

**Key observations:** Near-flat for α < 0.20 (~10.09-10.15 µg/m³) - the model is insensitive to moderate diffusion rates at the 1-hour horizon. Steep rise for α > 0.50 - excessive neighbor averaging washes out local temporal autocorrelation, degrading prediction quality. The optimal α = 0.113 sits at the lower end, confirming that 1-hour spatial diffusion is slow.

---

### Figure: task7_propagation_model.png

![Diffusion Model vs Observed NO2](figures/task7_propagation_model.png)

**What this shows:** Two panels for 200 test hours at an example sensor. Top: observed NO2 (black), persistence baseline (blue), 1-step diffusion (dashed orange), recursive diffusion (dotted purple). Bottom: residuals from both diffusion variants.

**Key observations:**
- Top: 1-step diffusion closely tracks observed NO2, with a marginal improvement over persistence on individual spikes. Recursive diffusion drifts to the network mean within 20-30 hours.
- Bottom: 1-step residuals (orange) fluctuate symmetrically around zero - no directional bias. Recursive residuals (purple) grow monotonically as the simulation diverges from reality.

---

### Figure: task7_mae_map.png

![Per-Sensor MAE Map](figures/task7_mae_map.png)

**What this shows:** Sensor locations coloured by test-period MAE (yellow = low, red = high).

**Key observations:** A spatial gradient - park and suburban sensors in northwest/southwest (yellow/light) show low errors. Central/peripheral traffic-adjacent sensors (red) show high errors. This spatial structure confirms that the diffusion model is most accurate where pollution is driven by broad background conditions and least accurate where hyperlocal emission sources create unpredictable episodic spikes.

---

## 12. Task 8 - Parallelization

### Goal

For every active (year, sensor) pair for the selected pollutants, compute a correlation matrix of all simultaneously measured variables. Demonstrate parallel speedup, verify correctness, and use the matrices to identify stable variable-pollutant associations.

### Why Per-Year, Per-Sensor Correlation Matrices

The project brief requires computing "for each year and each sensor, the hourly correlation matrix." This design:
1. **Year-specific:** tracks whether associations are stable or shifting (e.g., does traffic-NO2 correlation change after the Low Emission Zone in 2018?).
2. **Sensor-specific:** reveals spatial heterogeneity (sensors near parks vs. motorways may show different meteorology-pollution associations).
3. **Hourly resolution:** captures the short-term, within-day co-variation most directly relevant to emission mechanisms.

### Why ProcessPoolExecutor (Not ThreadPoolExecutor)

Python's GIL prevents multiple threads from executing Python bytecode simultaneously - making multithreading ineffective for CPU-bound tasks like matrix computation. `ProcessPoolExecutor` spawns true separate OS processes, each with its own interpreter and memory, bypassing the GIL entirely.

### Why Chunked Per-Year Reading (USE_PARTITIONS=True)

Without partitioning, each of 471 workers would scan the full 64M-row dataset to find its (year, sensor) subset: 471 × 64M = 30 billion row reads. With pre-split year files, each worker reads only its ~2.7M-row year slice: 471 × 2.7M = 1.3 billion row reads - a **23× I/O reduction**. This is the dominant optimization for the full dataset.

### Terminal Output

```
Job scope: 471 pollutant-active year×sensor jobs
  (full cross-product: 576 = 24 years × 24 sensors)
Using pre-partitioned year files.

Sequential: 471 completed | Time: 1,679.63 s (≈28 min)
Parallel (2 workers): 471 completed | Time: 896.10 s (≈15 min)
Speedup: 1.87×

Correctness check: all 471 matrix hashes match ✓
Valid variable pairs: 68,073
Pairs < 336 overlapping hours: 0
Mean % pairs with |corr| > 0.6: 39.2%
```

**Why speedup is 1.87× instead of ideal 2.0×:**
- I/O contention: two workers competing for the same file system bandwidth
- Process-spawn overhead: forking processes has a fixed cost
- Cache warming: parallel run may benefit from OS file-system cache warmed by sequential run (explicitly flagged in output)

**Top sensors by average inter-variable correlation:**

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Sensor</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">avg_mean_corr</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Interpretation</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Ramón y Cajal</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">0.628</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Central arterial - strong co-variation between traffic, NO2, meteorology</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Plaza de España</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">0.543</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Central pedestrian hub</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Cuatro Caminos</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">0.527</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Major intersection</td>
  </tr>
  </tbody>
</table>
### Pollution Associations

```
[NO2] Strongest positive (traffic -> more NO2):
  OC_KRIGING (occupancy/congestion):  r = +0.222, 100% positive across jobs
  TI_KRIGING (traffic intensity):     r = +0.230, 99.1% positive
  SP_KRIGING (average speed):         r = +0.218, 97.5% positive

[NO2] Strongest negative (meteorology -> less NO2):
  VV (wind speed):    r = -0.390, 100% negative <- strongest consistent effect
  TEMP:               r = -0.258, 100% negative
  RS (solar rad.):    r = -0.234, 100% negative

[O3] Strongest positive (meteorology -> more O3):
  TEMP:    r = +0.633, 100% positive <- classic photochemical signature
  RS:      r = +0.520, 100% positive

[O3] Strongest negative:
  HR (humidity):  r = -0.642, 100% negative <- hot/dry conditions -> max O3
```

**Physical interpretation of key associations:**
- **Wind speed -> NO2 (r = −0.39, 100%)**: Stronger winds dilute and transport NO2 away. The most consistent negative association in the entire dataset - holds for every sensor in every year.
- **Congestion -> NO2 (r = +0.22, 100%)**: Stationary vehicles in congested traffic produce more combustion emissions per km than free-flowing traffic. Consistent with stop-and-go driving increasing per-vehicle NOx output.
- **Temperature -> O3 (r = +0.63, 100%)** + **Humidity -> O3 (r = −0.64, 100%)**: The classical photochemical O3 signature. Hot, dry conditions maximize UV irradiance and ozone formation rates.

---

### Figure: task8_runtime.png

![Task 8 Runtime Comparison](figures/task8_runtime.png)

Sequential (1679.6 s) vs. parallel (896.1 s) wall-clock times with the 1.87× speedup clearly visible.

---

### Figure: task8_pollution_associations.png

![Pollution Variable Associations Heatmap](figures/task8_pollution_associations.png)

**What this shows:** Heatmap of mean hourly correlations between meteorological/traffic variables (rows, stable across ≥5 jobs) and the four selected pollutants (columns). Blue = negative association, red = positive.

**Key observations:**
- VV row: Strongly blue for NO2 and SO2 (wind disperses combustion pollutants), moderately red for O3 (Atlantic wind brings background O3).
- TEMP row: Red for O3, blue for NO2 - classical photochemical vs. thermal inversion trade-off.
- RS row: Same pattern as TEMP (UV drives both photolysis and O3 formation).
- HR row: Strongly blue for O3 (humid = cloudy = less UV = less O3).
- Traffic rows (TI, OC): Moderate red for NO2, near-neutral for O3 (traffic NO destroys O3 via titration, creating a counteracting negative effect).

---

## 13. Task 9 - Forecasting Model *(Optional)*

### Goal

Predict city-level NO2 concentration 24 hours ahead using lagged meteorological and traffic predictors. Evaluate performance, and interpret which variables are most predictive.

### Why 24 Hours as the Forecast Horizon

1. **Operational utility:** Health authorities need at least 24h warning to issue advisories or activate emergency protocols.
2. **Predictive signal:** NO2 has strong 24-hour periodicity - same-time-yesterday is a meaningful predictor.
3. **Feature availability:** Lagged meteorological observations (1h, 6h, 24h back) are all available at forecast issue time. No future data is needed.

### Why City-Level Spatial Collapse

Spatial variation is deliberately collapsed to city-level means for this model:
1. **Interpretability:** One model produces clean city-wide interpretations. 24 sensor-specific models would require 24× the interpretation effort.
2. **Data availability:** Meteorological predictors are only measured at a subset of stations - city-level means pair them cleanly with city-level NO2.
3. **Baseline role:** This model serves as an interpretable baseline. Sensor-specific spatial models could be built on top in future work.

### Why This Is a Genuine Causal Forward Forecast

All predictors use lagged values only - NO2 at lags 1h/3h/6h/24h, meteorological and traffic variables at the same lags. The target is NO2 at t+24h. No concurrent or future values are used anywhere. This is an operationally valid real-time forecast setup.

### Why Ridge AND Random Forest

**Ridge (L2-regularized linear model):**
- Standardized inputs -> coefficient magnitudes directly comparable across features
- Interpretable: coefficient sign = direction of effect, magnitude = relative importance
- L2 penalty prevents overfitting when correlated lag features compete (e.g., NO2_lag_1h and NO2_lag_3h are highly correlated)

**Random Forest (80 trees, max depth 10):**
- Captures nonlinear interactions (e.g., high humidity AND calm wind together trap NO2 more than either alone)
- Feature importances provide a complementary view
- Expected to outperform Ridge if true predictive relationship is nonlinear

### Terminal Output

```
Effective issue window: 2019-01-02 -> 2024-01-01
Usable forecast rows: 43,801 (20.8% of 210,384 calendar hours)

24h-ahead model comparison:
          model     RMSE      MAE      R²
Persistence_24h   16.617   11.286   0.288
          Ridge   15.525   11.853   0.378
   RandomForest   14.526   10.696   0.456  <- best
```

Random Forest achieves R² = 0.456 vs. persistence R² = 0.288 - a 58% improvement in explained variance. The modest absolute R² (≤0.46) is expected at the 24-hour horizon: without actual weather forecast inputs (only lagged observations), the model cannot anticipate meteorological shifts.

### Top Ridge Coefficients

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Feature</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Coefficient</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Physical interpretation</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">NO2_lag_1h</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">+9.83</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Strong autocorrelation - recent NO2 predicts next-day NO2</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">TEMP_lag_3h</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">+9.50</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Captures time-lag temperature pattern</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">TEMP_lag_1h</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">−6.49</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Immediate temperature negative relationship</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">TI_IDW_lag_1h</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">+5.88</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Recent traffic -> higher tomorrow's NO2</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">RS_lag_3h</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">−4.21</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Recent solar radiation -> photolysis, lower NO2</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">NO2_lag_24h</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">+3.32</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Same-time-yesterday predictor</td>
  </tr>
  </tbody>
</table>
### Top Random Forest Importances

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Feature</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Importance</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Physical interpretation</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">NO2_lag_1h</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">0.530</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Dominant - strong temporal autocorrelation</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">PRE_lag_1h</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">0.045</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Anticyclonic high pressure -> trapping</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">RS_lag_6h</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">0.037</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Solar radiation -> photolysis</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">NO2_lag_24h</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">0.030</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">24h periodicity</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">target_weekday_sin</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">0.028</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Monday-Friday traffic cycle</td>
  </tr>
  </tbody>
</table>
The Random Forest assigns 53% of importance to `NO2_lag_1h` - recent NO2 concentration is by far the best predictor of future NO2. Atmospheric pressure is second-most important (anticyclones suppress vertical mixing). Calendar features capture the weekly traffic cycle.

**Important caveat:** These are *predictive associations*, not causal effects. A positive coefficient on traffic intensity means traffic history is a useful predictor of NO2 - not that reducing traffic by 1 unit would reduce NO2 by the coefficient amount.

---

### Figure: task9_24h_forecast.png

![24h NO2 Forecast](figures/task9_24h_forecast.png)

**What this shows:** 400-hour slice from the test period. Actual NO2 (black), 24h persistence baseline (green), Ridge forecast (orange), Random Forest forecast (red).

**Key observations:** All three forecasts capture multi-day pollution elevation and recovery periods. Random Forest (red) tracks the observed trace most closely, particularly on moderate episodes. All models under-predict sharp short-duration spikes - episodic exceedances driven by unusual meteorological events (strong inversions, specific wind directions) are inherently hard to forecast 24h ahead with lagged observations only.

---

## 14. Task 10 - Final Visualization

### Goal

Produce consolidated presentation-ready summary panels communicating the key findings in a format suitable for a 10-minute presentation (8-15 slides).

---

### Figure: task10_panel_A_temporal.png - Temporal Patterns

![Panel A Temporal](figures/task10_panel_A_temporal.png)

**What this shows:** Top: City-wide NO2 monthly trend 2001-2024 (imputed series). Bottom: Seasonal bar charts (month 1-12) for all four pollutants.

**Why this is Panel A (the opening):** The long-run NO2 decline and the seasonal cycle are the most policy-relevant findings. A non-expert audience should immediately grasp: (1) NO2 has declined ~44% over 24 years - emission policy is working; (2) there is a strong and well-understood seasonal cycle; (3) O3 and NO2 show opposite seasonal patterns, which has important implications for ozone management.

---

### Figure: task10_panel_B_networks.png - Network Views

![Panel B Networks](figures/task10_panel_B_networks.png)

**What this shows:** Left: Spatial KNN k=2 network with nodes at UTM coordinates, coloured by degree. Right: NO2 sensor-sensor Pearson correlation heatmap.

**Why this dual panel is powerful:** The juxtaposition directly illustrates the central message of Tasks 5-6. The sparse spatial network (32 edges, meaningful community structure) represents geographic proximity. The dense correlation heatmap (nearly all cells dark green) represents behavioral synchrony. The contrast shows that while sensors have defined local neighborhoods, their NO2 dynamics are synchronized city-wide - supporting the interpretation that city-scale drivers (meteorology, traffic rhythms, heating season) dominate over local microenvironments for NO2.

---

### Figure: task10_panel_C_diurnal_quality.png - Diurnal Profiles & Data Quality

![Panel C Diurnal Quality](figures/task10_panel_C_diurnal_quality.png)

**What this shows:** Left: Diurnal pollution profiles (hour 0-23) for all four pollutants, with morning and evening rush windows shaded. Right: Top-20 variables by missingness rate (horizontal bar chart).

**Why combine these two:** They answer the question "what do we know and how confidently?" together. The diurnal profiles show well-characterized daily cycles - high confidence. The missingness chart shows where that confidence is lower: meteorological variables at 80%+ missing should be interpreted cautiously; air quality variables at 1-5% missing are highly reliable.

**Key diurnal observations:** NO2 double-peak precisely at rush hours. O3 midday maximum. The strong anti-correlation between NO2 and O3 profiles is visible in a single glance - when NO2 peaks (rush hours), O3 is low (consumed by NO); when NO2 troughs (midday photolysis), O3 peaks.

---

## 15. Global Design Decisions & Constants

### Why NO2, O3, PM10, SO2

These four satisfy four simultaneous criteria:
1. **Policy relevance:** Primary pollutants regulated by EU Air Quality Directive (2008/50/EC) and its 2024 revision.
2. **Full temporal coverage:** All four span 2001-2024, enabling uninterrupted 24-year analyses.
3. **Diverse chemistry:** Combustion direct emission (NO2), photochemical secondary (O3), mixed primary/secondary particulate (PM10), sulfurous combustion (SO2). Together they capture the full range of atmospheric emission regimes.
4. **Spatial density:** All measured at 10-24 of 24 stations.

### Why CORR_THR = 0.60

0.60 is a standard meaningful-correlation threshold in environmental science. At r = 0.60, shared variance is r² = 0.36 - the two variables share 36% of their variance, enough to constitute a meaningful association. Below 0.60, shared variance < 36% and the association is dominated by other factors. The threshold is also a round number that is easy to communicate.

### Why min_periods = 168 (Task 8)

168 hours = exactly 1 week of hourly data. A Pearson correlation from fewer than 168 points is unreliable - with < 20 observations, r > 0.5 can easily arise by chance. One week covers the basic weekly traffic rhythm (5 weekdays + 2 weekend days), capturing the dominant short-term co-variation pattern.

### Why gc.collect() After Each Pollutant

Wide pivot tables (time × sensor) for the full 24-year, 24-sensor dataset reach ~10-50 MB per pollutant in float32. Across 4 pollutants, these accumulate to ~200 MB if not explicitly freed. Python's garbage collector does not immediately reclaim large numpy arrays when they go out of scope. Explicit `gc.collect()` calls keep peak RSS manageable on machines with < 8 GB available RAM.

---

## 16. Scalability & Reproducibility

### Memory Scaling

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Stage</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Peak RAM</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Dataset load (64M rows, efficient dtypes)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">~2.0 GB</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Task 3 pivot per pollutant</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">~200 MB</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Task 7 hourly pivot (24 sensors × 210k hours)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">~400 MB</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">Task 8 worker (year-partitioned, chunked read)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">< 50 MB</td>
  </tr>
  </tbody>
</table>
### Determinism

- Random masking in Task 3: fixed seeds 42 and 43.
- Train/test splits: strictly chronological, no shuffling.
- Community detection: greedy modularity optimization (deterministic given same input).

### Expected Full-Dataset Runtime (2 CPU cores, SSD)

<table style="border-collapse:collapse;width:100%;background:#ffffff;margin:10px 0;">
  <thead><tr>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Task</th>
    <th style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;font-weight:bold;text-align:left;">Approx. time</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">1 (Load + Inspect)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">5-10 min</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">2 (Missingness + quality)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">15-20 min</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">3 (Imputation)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">25-40 min</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">4 (Temporal analysis)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">5-10 min</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">5 (Spatial network)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">< 1 min</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">6 (Correlation network)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">3-5 min</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">7 (Propagation model)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">10-15 min</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">8 (Parallelization)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">15 min parallel</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">9 (Forecasting)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">5-8 min</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">10 (Final plots)</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">< 1 min</td>
  </tr>
  <tr>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">**Total**</td>
    <td style="background:#ffffff;padding:6px 14px;border:1px solid #cccccc;text-align:left;">**~60-90 min**</td>
  </tr>
  </tbody>
</table>
### Citation

[1] David María-Arribas et al. "METRAQ Air Quality dataset." Hugging Face, 2024.
https://huggingface.co/datasets/dmariaa70/METRAQ-Air-Quality
