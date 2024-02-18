 # Hourly electricity consumption forecasts for the following day based on Time Series Imaging

![NNTS](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/1d9f9a2c-376e-4165-8ae3-6cd2f791bb5a)


This script implements a methodology that combines ***deep learning***, ***image and time series processing*** to ***predict hourly electricity consumption for the next day***.

The proposed methodology consists of 3 phases:

- Data preparation
- Conversion of time series to 2D images
- Training of Deep learning models
  
  ## Data preparation
  
  The first step consists of preparing a dataset in which each row represents the weekly hourly consumption (24x7) of any user and the cluster (cluster_ next_day) that best represents the next day's (24 hours) consumption for that user. This cluster is assigned by applying a clustering model to the 24 hours of consumption of the next day that will assign the day of consumption to one of 21 clusters. This model will be available for free.

<img width="993" alt="Captura de pantalla 2024-02-09 a las 17 04 01" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/a6a96b4c-3cb8-401a-b068-80e1ed09d749">

> **_NOTE:_** This script does not prepare the data as described, but the model for grouping days of hourly consumption into 21 centroids shown below is freely available in this repository, in case you are interested in completing the next day grouping column, using this model.

### Clustering model

![fig--k21cent-Ireland](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/9cb4ceef-1f5d-44c6-90a5-0ab72116627e)

   
## Time series imaging

Each time series can be represented by a curve as can be seen in the following example of a user's weekly consumption.

<img width="1466" alt="Captura de pantalla 2024-02-18 a las 10 52 35" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/d95f3253-9b4c-4f2d-8ec8-6b46ef72e7e3">

> In order to train machine learning models, each weekly consumption time series will be modelled as a 2D image, as will each of the 21 clusters in the model (one of these images will be the image representing the next day's consumption cluster). All this information will be needed for the supervised machine learning process carried out by the deep learning algorithms.

    

 ## Deep Learning with Time Series Imaging




## Authors ✒️

* **Francisco Rodríguez Gómez** 
* **José del Campo Ávila** 
* **Llanos Mora López** 

## License 📄

  This project is under GNU General Public License v3.0 [LICENSE.md](LICENSE.md) for more detail.

## Acknowledgements 🎁

  This work has been supported by the project RTI2018-095097-B-I00 at the 2018 call for I+D+i Project of the Ministerio de Ciencia, Innovación y Universidades, Spain. 
