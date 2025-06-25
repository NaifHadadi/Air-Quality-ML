### Pearson Correlation and Relationship Between Variables

We use the **Pearson Correlation Coefficient** to measure how strongly two variables are related. Specifically, it tells us:

- **How much they change together** (covariance),
- Relative to **how much they vary on their own** (standard deviation).

This gives a value between **–1 and +1**:
- **+1**: perfect positive linear relationship
- **0**: no linear relationship
- **–1**: perfect negative linear relationship

<br>

![image](https://github.com/user-attachments/assets/7c4f4664-f045-45fc-b0c3-0a0a07c814b2)

<br>
<br>

# Key Findings & Interpretation of the Heatmap

| **Variable Pair** | **Correlation** | **Strength & Interpretation**                            |
|-------------------|------------------|-----------------------------------------------------------|
| **NO2 – CO**      | **+0.89**        | Very strong positive correlation (shared emission sources) |
| **AWND – NO2**    | **–0.61**        | Strong negative correlation (wind disperses NO2)           |
| **AWND – CO**     | **–0.60**        | Strong negative correlation (wind disperses CO)            |
| **O3 – NO2**      | **–0.56**        | Strong negative correlation (NO2 scavenges ozone)          |
| **O3 – CO**       | **–0.49**        | Moderate-to-strong negative correlation                    |
| **TAVG – PM10**   | **+0.48**        | Moderate-to-strong positive correlation (temp ↔ dust/PM10) |
| **SO2 – NO2**     | **+0.53**        | Strong positive correlation (related pollution sources)    |




####  **1. Strong Coupling Between Combustion Pollutants (NO₂ and CO)**
- **Correlation:** +0.89
- **Implication:** NO₂ and CO are strongly correlated, indicating **common sources such as traffic and industrial combustion**. This confirms that **monitoring either can provide insights into general combustion-related pollution** in industrial zones.

####  **2. Wind Speed (AWND) Has a Major Dispersive Effect on Air Pollutants**
- **Correlations:**
  - AWND–NO₂: **–0.61**
  - AWND–CO: **–0.60**
  - AWND–PM2.5: **–0.34**
- **Implication:** Higher wind speeds are **consistently associated with reduced concentrations** of pollutants. This suggests that **wind acts as a natural ventilator**, and should be integrated into spatiotemporal pollutant dispersion models.

####  **3. Temperature Positively Influences Ozone and Particulate Levels**
- **Correlations:**
  - TAVG–O₃: **+0.38**
  - TAVG–PM10: **+0.48**
- **Implication:** Rising temperatures are associated with **increased ozone and coarse particulate levels**, reinforcing concerns about **climate change exacerbating air pollution** in heat-prone regions.

####  **4. Ozone Behavior is Inversely Related to Combustion Emissions**
- **Correlations:**
  - O₃–NO₂: **–0.56**
  - O₃–CO: **–0.49**
- **Implication:** **High NO₂ and CO levels suppress ozone concentrations**, likely due to chemical scavenging. This is critical when interpreting satellite-detected ozone concentrations in polluted environments.



