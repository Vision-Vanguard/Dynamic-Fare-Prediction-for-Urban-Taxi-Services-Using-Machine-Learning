# NYC-Taxi-Fare-Prediction

[<image-card alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" ></image-card>](https://opensource.org/licenses/MIT)
[<image-card alt="Python" src="https://img.shields.io/badge/Python-3.12-blue.svg" ></image-card>](https://www.python.org/)

## Project Overview

This repository contains the code for predicting dynamic taxi fares in New York City using machine learning, developed as part of the "(FINAL) Tech Olympics 2025: AI - Data Processing" competition. The goal is to build a predictive model that estimates the total fare (`total_price` in USD) for taxi trips based on historical data from 2016. The project demonstrates advanced data processing, feature engineering, and ensemble modeling techniques to achieve high accuracy on a regression task.

**Academic Problem Statement**:  
In urban transportation systems, accurate fare prediction is essential for optimizing pricing strategies, enhancing user experience, and improving operational efficiency. This study addresses the challenge of developing a dynamic pricing model for taxi services in New York City, based on historical trip data from 2016. The dataset includes features such as trip timestamps, passenger counts, pickup/dropoff coordinates, and storage flags, with the target variable being the total fare in USD.

The objective is to design a predictive system that minimizes pricing errors by leveraging advanced data processing techniques, feature engineering, and ensemble machine learning models. Key research questions include: (1) How can spatiotemporal features (e.g., distances, bearings, and cyclic time encodings) improve prediction accuracy? (2) What is the impact of clustering locations and handling temporal dependencies via time-based cross-validation? (3) How do gradient boosting methods compare to linear models in this regression task?

The model must achieve a root mean squared error (RMSE) below a threshold (e.g., equivalent to an R² > 0.5) on unseen data, demonstrating robustness for real-world deployment. This work contributes to the field of AI-driven urban mobility by providing an open-source implementation evaluated on a large-scale dataset of over 1.2 million training samples.

## Data

- **Source**: NYC Taxi Trip Data (2016) from the competition dataset.
- **Files**:
  - `train.csv`: 1,255,094 rows with 11 features (including `total_price` as target).
  - `test.csv`: 203,550 rows with 10 features (predictions required).
- **Key Features**: Pickup/dropoff datetime, latitudes/longitudes, passenger count, store_and_fwd_flag, trip duration.
- **Target**: `total_price` (fare in USD, skewed distribution; log-transformed during training).

Data statistics: Mean fare ~$22.43, Std ~$48.11, Range [$4.51, $29,411.82].

## Methods and Algorithms

### Feature Engineering
- Distance: Haversine and Manhattan formulas.
- Direction: Bearing calculation.
- Time: Cyclic sine/cosine encoding for hours/minutes; flags for rush hours, weekends, nights, holidays.
- Clustering: K-Means (50 clusters) on location coordinates.
- Other: Speed (distance/duration), day parts.

### Models
- Linear: Ridge Regression, Linear Regression.
- Boosting: LightGBM, CatBoost, XGBoost.
- Ensemble: Weighted averaging by CV scores; Stacking with Ridge meta-learner.

### Validation
- Time-based 5-fold CV (monthly splits).
- Metrics: RMSE and custom exponential score.

## Results

- **Ridge**: RMSE ≈ 3.23e+29, Score = 0.00 (baseline).
- **LightGBM**: RMSE ≈ 39.59, Score = 43.91.
- **Ensemble**: Improves overall RMSE by blending models.

Final predictions clipped to non-negative and rounded to 2 decimals.

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/NYC-Taxi-Fare-Prediction.git
cd NYC-Taxi-Fare-Prediction
pip install -r requirements.txt  # Create this file with: pandas numpy scikit-learn lightgbm catboost xgboost joblib
