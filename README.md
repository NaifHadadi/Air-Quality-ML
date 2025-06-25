# LA-Air-Quality-ML-Analysis-Predictive-Modeling-Pollution-Insights
Data-driven analysis and machine learning models for forecasting air pollution levels in Los Angeles using historical weather and pollutant concentration data. Includes performance evaluation, feature insights, and predictive reliability assessment.
<br>

# Data Details

The dataset was compiled and preprocessed from multiple official sources. The details of the data processing can be found in the [data processing guide](/docs/data_details.md).  

<br>

### **The Processed Data Structure:**

| Column                                 | Description                                   |
|---------------------------------------|-----------------------------------------------|
| `O3`                                  | Daily mean ozone concentration (ppm)          |
| `CO`                                  | Daily mean carbon monoxide (ppm)              |
| `SO2`                                 | Daily mean sulfur dioxide (ppb)               |
| `NO2`                                 | Daily mean nitrogen dioxide (ppb)             |
| `O3 AQI`, `CO AQI`, `SO2 AQI`, `NO2 AQI` | AQI values corresponding to O₃, CO, SO₂, NO₂ respectively |
| `PM2.5`                               | Daily mean PM2.5 concentration (µg/m³)        |
| `PM10`                                | Daily mean PM10 concentration (µg/m³)         |
| `AWND`                                | Average wind speed (mph)               |
| `PRCP`                                | Total precipitation (inches)            |
| `TAVG`                                | Daily average temperature (°C)          |
| `year`                                | Year extracted from the date                  |
| `month`                               | Month extracted from the date (1–12)          |
| `season`                              | Meteorological season (e.g., Winter, Summer)  |


<br>

### **Temperature Analysis:**

<img src="https://github.com/user-attachments/assets/9f7625b0-ac29-43d8-934b-f804cfc58a42" width="800">

<img width="800" alt="image" src="https://github.com/user-attachments/assets/dbd5e7fc-fce8-45eb-be44-c3d11997068d" />

<br>
<br>

**Los Angeles Temperature Overview (°C):**

Based on official data sources, typical seasonal temperatures in Los Angeles are as follows:

- **Warmest Months (July–September):**  
  - **Average High:** ~28.5 °C  
  - **Average Low:** ~18.5 °C

- **Coolest Months (December–February):**  
  - **Average High:** ~20.0 °C  
  - **Average Low:** ~9.5 °C

These values are broadly consistent with the temperature trends observed in our dataset. The seasonal pattern confirms that **summer is the warmest**, followed by **fall**, then **spring**, and **winter** as the coolest season.

**Reference:**  
Climate normals for Los Angeles based on NOAA 1991–2020 data, available at [NOAA National Centers for Environmental Information](https://www.ncei.noaa.gov/)

![newplot1](https://github.com/user-attachments/assets/5854e4fd-9f29-4aff-9d52-2de8617ba1d8)

<br>

# Correlation Study
The main finding of the correlation study is summarized in the following table (for more details, refer to the [correlation analysis document](/docs/correlations.md) ):
Key Findings & Interpretation of the Heatmap

<br>

| **Variable Pair** | **Correlation** | **Strength & Interpretation**                            |
|-------------------|------------------|-----------------------------------------------------------|
| **NO2 – CO**      | **+0.89**        | Very strong positive correlation (shared emission sources) |
| **AWND – NO2**    | **–0.61**        | Strong negative correlation (wind disperses NO2)           |
| **AWND – CO**     | **–0.60**        | Strong negative correlation (wind disperses CO)            |
| **O3 – NO2**      | **–0.56**        | Strong negative correlation (NO2 scavenges ozone)          |
| **O3 – CO**       | **–0.49**        | Moderate-to-strong negative correlation                    |
| **TAVG – PM10**   | **+0.48**        | Moderate-to-strong positive correlation (temp ↔ dust/PM10) |
| **SO2 – NO2**     | **+0.53**        | Strong positive correlation (related pollution sources)    |

<br>

![image](https://github.com/user-attachments/assets/b2fb5adb-baf7-40f5-8606-3f1ed547c479)


<br>

**The key takeaways from this study are:**

####  **1. Strong Coupling Between Combustion Pollutants (e.g, NO₂ and CO)**
####  **2. Wind Speed (AWND) Has a Major Dispersive Effect on Air Pollutants**
- **Correlations:**
  - AWND–NO₂: **–0.61**
  - AWND–CO: **–0.60**
  - AWND–PM2.5: **–0.34**
####  **3. Temperature Positively Influences Ozone and Particulate Levels**
####  **4. Ozone Behavior is Inversely Related to Combustion Emissions**

<br>

---

# ML Modeling
In this project, we used both a Multi-Layer Perceptron (MLP) and an XGBoost Regressor (XGBRegressor) to build predictive models, focusing specifically on **NO₂**, one of the most critical air pollutants.

### MLP Architecture

The MLP is a 4-layer feedforward neural network implemented using PyTorch.
**Architecture Details**:
- **Input layer**: size = `input_shape`
- **Hidden Layer 1**: `Linear(input_shape → hidden_units)` + ReLU + Dropout(0.2)
- **Hidden Layer 2**: `Linear(hidden_units → hidden_units)` + ReLU + Dropout(0.2)
- **Hidden Layer 3**: `Linear(hidden_units → hidden_units)` + ReLU
- **Output layer**: `Linear(hidden_units → output_shape)`
  
<br>

## Models Performance:
| Model Name           | MAE    | R² Score | RMSE   |
|----------------------|--------|----------|--------|
| Neural Network (NN)                 | 2.3180 | 0.8760   | 3.0079 |
| NN with L2 Penalty (Ridge)          | 2.3346 | 0.8740   | 3.0321 |
| NN with L1 Penalty (Lasso)          | 2.3354 | 0.8735   | 3.0381 |
| NN with L1 + L2 (ElasticNet)        | 2.3118 | 0.8766   | 3.0017 |
| **XGBoost Regressor**               | **2.2481** | **0.8851**   | **2.8855** |


#### Key Observations:
- The **XGBoost Regressor** outperformed all neural network variants in terms of **lowest MAE & RMSE, and highest R²**.
- Among the neural network models, the **L1+L2 (ElasticNet)** regularization provided the best performance.
- Adding L1 or L2 penalties slightly reduced performance compared to the base MLP, suggesting limited overfitting in the unregularized model.
- **MAE and RMSE are close in all models**, indicating:
  - Extreme outliers do not heavily skew the errors.
  - Most predictions are reasonably close to the true values, which enhances the **reliability and interpretability** of the reported performance metrics.

<br>

## XGBoost Performance Visualization: Actual vs Predicted, Residuals, and Distribution

The XGBoost model shows good predictive performance for NO₂ levels, with most points clustered near the ideal prediction line as shown in the plots below:

<img width="800" alt="image" src="https://github.com/user-attachments/assets/b2ed08bd-1aa5-4315-8062-e55d8d6fb871" />

<br>
<br>

**Key observations:**
- Best performance in mid-range concentrations (5-30 units)
- There are some important **Patterns**: 
  - Slight under-prediction at high concentrations (>30)
  - Some over-prediction at low concentrations (<5)

The residual plot demonstrates **homoscedasticity** (consistent error variance across predicted values), indicating our model handles variance well. The observed overprediction//underprediction at high/low concentrations is likely due to:
- Data scarcity in extreme ranges
- Fewer training samples for high/low-concentration patterns
