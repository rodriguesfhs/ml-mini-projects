# Research Centre Infrastructure Clustering with K-Means

A compact unsupervised machine learning project that groups research centres into quality tiers based on internal facilities and surrounding infrastructure, using K-Means clustering on tabular data.

## Overview

The dataset contains numerical features describing research centre capacity and the healthcare and facility landscape within 10 km of each centre. Because no ground-truth quality labels exist, the tiers are inferred from patterns in the data rather than predefined categories.

The resulting clusters are mapped to three interpretable tiers: **Basic**, **Standard**, and **Premium**.

## Approach

- Exploratory data analysis across infrastructure and access features, by centre and by city
- Feature selection based on domain relevance, with correlation analysis to identify redundancy
- Elbow method to justify the choice of k = 3 before model fitting
- Single-pipeline fit combining `StandardScaler` and `KMeans`, used as the sole source of labels and cluster profiles throughout
- Programmatic tier mapping derived from cluster centres in scaled feature space, robust to variation in cluster label assignment across runs
- Cluster evaluation via silhouette score
- Interpretation of tier distribution by city and the relative contribution of diversity vs density

## Features

| Feature | Description |
|---|---|
| `internalFacilitiesCount` | Number of internal facilities; direct proxy for centre capacity |
| `hospitals_10km` | Hospitals within 10 km |
| `pharmacies_10km` | Pharmacies within 10 km |
| `facilityDiversity_10km` | Variety of external facility types in the surrounding area |
| `facilityDensity_10km` | Density of external facilities within 10 km |

Geographic coordinates were excluded as they describe location rather than quality.

## Tools

- Python
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn

## How to run

```bash
git clone https://github.com/rodriguesfhs/ml-mini-projects
cd ml-mini-projects/research-center-assignment
pip install -r requirements.txt
jupyter notebook EDA_and_Model.ipynb
```

## Limitations

- Quality tiers are inferred, not ground-truth labels; the cluster-to-tier mapping is interpretive
- Correlated features (internal count, density, diversity) slightly over-weight their shared variance in the Euclidean distance K-Means optimises
- k = 3 is supported by the elbow plot and domain framing, but alternative values of k were not extensively evaluated
- A larger or more diverse dataset would allow more robust cluster validation
- The dataset is not included in this repository as it is proprietary. To run the notebook, substitute a CSV with the column structure described above.
