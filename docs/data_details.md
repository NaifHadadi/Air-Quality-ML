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