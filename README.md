# ExtremePrecipit
**High-performance analysis and visualization of extreme precipitation events**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Streamlit](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Data: Zarr/Dask](https://img.shields.io/badge/storage-Zarr%2FDask-orange.svg)](https://zarr.readthedocs.io/)

---

## Overview

**ExtremePrecipit** is a robust scientific platform engineered for the end-to-end analysis of extreme meteorological events. by combining high-performance data processing with advanced statistical modeling, it enables researchers to bridge the gap between raw NetCDF datasets and actionable climate insights.

Whether analyzing observed station data or high-resolution model outputs (AROME), the project provides a seamless transition from raw binary formats to interactive spatial visualizations and generalized extreme value (GEV) distribution modeling.

### Key capabilities

*   **High-throughput processing**: leverages **Zarr**, **Dask**, and **Polars** to handle large-scale multidimensional meteorological datasets with minimal memory overhead.
*   **Advanced statistical modeling**: automated fitting of **generalized extreme value (GEV)** distributions with support for non-stationary trends and likelihood ratio testing (LRT).
*   **Interactive spatial analytics**: a rich **Streamlit** dashboard for exploring return levels, return periods, and spatial trends across diverse temporal scales.
*   **Modular pipeline architecture**: fully automated workflows for data ingestion, cleaning, statistics extraction, and model selection.

---

## Technical stack

| Category | Technologies |
| :--- | :--- |
| **Core** | `Python 3.8+` |
| **Data engine** | `Xarray`, `Zarr`, `Dask`, `Polars`, `Numpy`, `Pandas` |
| **Analytics** | `Scipy`, `GEV adjustment`, `Likelihood ratio testing` |
| **Visualization** | `Streamlit`, `Plotly`, `Matplotlib`, `Geopandas`, `Pydeck` |
| **Format handling** | `NetCDF4`, `Parquet`, `YAML` |

---

## Installation

> [!TIP]
> Using a virtual environment is highly recommended to maintain clean dependencies.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/NCSdecoopman/ExtremePrecipit.git
    cd ExtremePrecipit
    ```

2.  **Install base dependencies**:
    ```bash
    pip install streamlit huggingface_hub numpy pandas xarray zarr dask \
                geopandas shapely pydeck plotly matplotlib pyyaml netCDF4 tqdm polars
    ```

---

## Quick start

### 1. Data ingestion
Retrieve the pre-processed datasets and metadata from Hugging Face:
```bash
python download_data.py
```

### 2. Execute pipelines
Run the full suite of processing steps automatically:
```bash
python src/pipelines/run_all.py
```
*alternatively, you can run individual pipelines located in `src/pipelines/` for specific tasks like Zarr conversion or GEV fitting.*

### 3. Launch the dashboard
Start the interactive visualization environment:
```bash
streamlit run main.py
```

---

## Project structure

```text
ExtremePrecipit/
├── main.py                 # Streamlit application entry point
├── download_data.py        # automated data acquisition script
├── config/                 # global pipeline configuration (YAML)
├── data/                   # data repository (raw, Zarr, statisticals, GEV)
├── logs/                   # detailed pipeline execution logs
├── src/                    # core technical implementation
│   ├── pipelines/          # processing workflows (ETL, statistics, modeling)
│   ├── modules/            # scientific calculation & visualization logic
│   └── utils/              # infrastructure helpers (logging, config, data I/O)
└── app/                    # Streamlit-specific modules and internal config
```

---

## Methodology

The project follows a rigorous scientific workflow:
1.  **Conversion**: NetCDF files are transcoded to **Zarr** for cloud-optimized, chunked access.
2.  **Statistics**: annual maxima are extracted and stored in high-performance **Parquet** files.
3.  **Modeling**: GEV distributions are fitted to the series, with model selection based on statistical significance (LRT).
4.  **Validation**: model outputs are benchmarked against observed station data to ensure fidelity.

---

## Contribution

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. any contributions you make are **greatly appreciated**.

---

## License

Distributed under the **MIT License**. see `LICENSE` for more information.
