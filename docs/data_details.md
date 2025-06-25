### Data Sources and Descriptions

1. **pm10 & PM2.5**: Contains daily PM_X (particulate matter) concentration data:  
   - Source: U.S. Environmental Protection Agency (EPA) Air Quality System (AQS) 
      - Link: https://www.epa.gov/outdoor-air-quality-data/download-daily-data
   - Temporal coverage: Daily  
   - Data Shape: (4134, 21) 

2. **data_wth**: Contains daily weather observations from meteorological stations:  
   - Source: National Centers for Environmental Information (NCEI) 
      - Link: https://www.ncei.noaa.gov/cdo-web/datasets 
   - Temporal coverage: Daily  
   - Data Shape: (314376, 31)

3. **data_poll**: Contains ground-level air pollutant concentration data and corresponding AQI values:
   - Source: From 2000-2016 (https://www.kaggle.com/datasets/sogun3/uspollution/data) and from 2016 - 2023 they were obtained from multiple resources.   
   - Temporal coverage: Daily measurements  
   - Data Shape: (665414, 22)  


These datasets will be cleaned, spatially filtered, and aggregated to daily averages.
Finally, they will be merged based on the 'Date' column to produce a consistent and analysis-ready dataset.

---

### **Schema of the original datasets**

### daily PM10 & PM2.5
| pm10        |                                          |                                                  |
|-------------|------------------------------------------|--------------------------------------------------|
|             | Daily Mean PM10 Concentration            | PM10 measurement (µg/m³)                         |
|             | Daily AQI Value                          | AQI for PM10                                     |
|             | Site Latitude, Site Longitude            | Monitoring station coordinates                   |
|             | Site ID, Source, POC                     |  identifiers for the monitoring site    |
|             | Units, Local Site Name                   |                                                  |
|             | Daily Obs Count, Percent Complete        |                        |
|             | Method Code, AQS Parameter Description   |             |
|             | CBSA Code, CBSA Name                     |                |
|             | State FIPS Code, County FIPS Code        |                                                  |


### Weather observations from meteorological stations
| data_wth    |                                          | Station identifiers                              |
|-------------|------------------------------------------|--------------------------------------------------|
|             | LATITUDE, LONGITUDE, ELEVATION           | Geolocation and altitude                         |
|             | DATE                                     | Observation date                                 |
|             | AWND, PRCP, TAVG, TMAX, TMIN             | Weather features: wind, precipitation, temp      |
|             | SNOW, SNWD, TOBS                         | Snow and observed temperature data               |
|             | WT01–WT22                                | |




##### Pollutant concentration data
| data_poll   |                                          | Description                                      |
|-------------|------------------------------------------|--------------------------------------------------|
|             | Date                                     | Observation date                                 |
|             | Address, State, County, City            | Location metadata                                |
|             | O3 Mean, CO Mean, SO2 Mean, NO2 Mean     | Mean daily concentrations of pollutants          |
|             | O3 AQI, CO AQI, SO2 AQI, NO2 AQI         | Corresponding Air Quality Index values           |
|             | O3 1st Max Value, CO 1st Max Value, etc. |       |
|             | O3 1st Max Hour, etc.                    |              |


---

The goal is to clean and filter the above data to produce a unified daily dataset for **Los Angeles**, containing the following variables:

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
