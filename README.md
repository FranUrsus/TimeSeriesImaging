 # Prediction of electrical energy consumption based on Time Series Imaging


This script implements a methodology that combines ***deep learning***, ***image and time series processing*** to ***predict hourly electricity consumption for the next day***.

The tool is very simple to use and consists of the following steps.

  ## Data preparation
  
  The first step consists of preparing a dataset in which each row represents the weekly hourly consumption (24x7) of any user and the cluster (cluster_ next_day) that best represents the next day's (24 hours) consumption for that user. This cluster is assigned by applying a clustering model to the 24 hours of consumption of the next day that will assign the day of consumption to one of 21 clusters. This model will be available for free.

<img width="993" alt="Captura de pantalla 2024-02-09 a las 17 04 01" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/a6a96b4c-3cb8-401a-b068-80e1ed09d749">

> **_NOTE:_** This script does not prepare the data as described, but the model for grouping days of hourly consumption into 21 centroids shown below is freely available in this repository, in case you are interested in completing the next day grouping column, using this model.

### Clustering model

![fig--k21cent-Ireland](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/9cb4ceef-1f5d-44c6-90a5-0ab72116627e)

   
## Time series imaging

Each time series can be represented by a curve as can be seen in the following example of a user's weekly consumption.

<img width="617" alt="Captura de pantalla 2024-02-09 a las 18 02 48" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/dcbc8e1c-9661-42bb-8cab-bced69d59784">

    

 4. **Deep Learning with Time Series Imaging** 




## Authors ✒️

* **Francisco Rodríguez Gómez** 
* **José del Campo Ávila** 
* **Llanos Mora López** 

## License 📄

  This project is under GNU General Public License v3.0 [LICENSE.md](LICENSE.md) for more detail.

## Acknowledgements 🎁

  This work has been supported by the project RTI2018-095097-B-I00 at the 2018 call for I+D+i Project of the Ministerio de Ciencia, Innovación y Universidades, Spain. 
