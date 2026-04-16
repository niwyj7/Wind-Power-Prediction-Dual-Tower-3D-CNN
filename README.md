# Wind-Power-Prediction-Dual-Tower-3D-CNN

# Reigional Wind Power Spatial Decomposition & Forecast (Dual Tower 3D CNN)

This repository contains a deep learning pipeline designed to forecast and spatially decompose wind power generation. It leverages geographical meteorological data (e.g., ECMWF weather data) divided into two regional grids (South and North).

## Project Aim & Background

The primary objective of this project is to **decompose the wind power generation of a region into its Northern and Southern grid components**. 

This initiative is driven by energy trading strategies. Market traders observed that the imbalance between north and south power generation might influence regional power prices. The goal here is to discover and extract these **latent regional generation signals**. 

Because there are no actual ground-truth targets for the strictly decomposed regional generation (only the aggregated total is available), traditional supervised validation for the sub-regions is impossible. Instead, the extracted latent representations from the dual-tower model are **validated practically by energy traders** to ensure they align with market realities and provide actionable trading insights.

## Core Architecture

The core model is a **Dual Tower 3D Convolutional Neural Network (CNN)** equipped with a **Channel Attention** mechanism to extract spatial-temporal features effectively.

* **Latent Feature Extraction:** The dual-tower design explicitly forces the network to learn independent representations for the North and South grids before fusing them, isolating the spatial imbalance.
* **Physically Constrained Outputs:** Ensures strictly non-negative wind power predictions using `Softplus` activation and post-processing bounds to match real-world physics.
* **Strict Time-Series Validation:** Prevents data leakage by strictly splitting training and validation sets chronologically (no random splits).
* **Data Alignment Verification:** Ensures strict index matching across multiple data sources before tensor creation.
* **Channel Attention:** Enhances feature extraction dynamically without overcomplicating the spatial dimensions.

<img width="1536" height="1024" alt="ChatGPT Image Apr 13, 2026, 09_33_56 PM" src="https://github.com/user-attachments/assets/09f6d79a-b783-4cb0-aade-361651a60dd6" />

## Check Consistency
Visualise weight heatmaps to check model convergence consistency/learning consistency (rolling train, daily model update)
Examples:
<img width="1050" height="750" alt="image" src="https://github.com/user-attachments/assets/a1173e51-11b0-4e68-85db-365b2c6b1c99" />

<img width="1050" height="750" alt="image" src="https://github.com/user-attachments/assets/eae85336-bc3d-4206-9f0e-b2aab658c3d2" />
