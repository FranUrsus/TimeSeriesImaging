 #  Deep learning and time series imaging for the next day electricity consumption forecasts

![NNTS](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/1d9f9a2c-376e-4165-8ae3-6cd2f791bb5a)


This script implements a methodology that combines ***deep learning***, ***image and time series processing*** to ***predict hourly electricity consumption for the next day***.

The proposed methodology consists of 3 phases:

- Data preparation
- Time series imaging
- Training of Deep learning models

## Data preparation
  
  The first step consists of preparing a dataset in which each row represents the weekly hourly consumption (24x7) of any user and the cluster (cluster_ next_day) that best represents the next day's (24 hours) consumption for that user. This cluster is assigned by applying a clustering model to the 24 hours of consumption of the next day that will assign the day of consumption to one of 21 clusters. This model will be available for free.

<img width="1008" alt="Captura de pantalla 2024-02-19 a las 19 03 33" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/c85737ec-c687-4f9b-bc32-c5b6d673bf31">

> **_NOTE:_** This script does not prepare the data as described, but the model for grouping days of hourly consumption into 21 centroids shown below is freely available in this repository, in case you are interested in completing the next day grouping column, using this model.

## Clustering model

The centroids of the model that has been used to assign the next day's cluster to each weekly consumption are as follows:

<img width="727" alt="centroids" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/d6e7827a-af06-4bf9-9f90-20e59d5a4962">

## Time series imaging

Time series can be represented as 2D curves representing the value for each instant in the time series, but this way of modelling time series is not the most suitable for training models with deep learning algorithms to make predictions from time series images.

### Week consumption

<img width="1466" alt="Captura de pantalla 2024-02-18 a las 10 52 35" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/d95f3253-9b4c-4f2d-8ec8-6b46ef72e7e3">

> In order to train machine learning models, each weekly consumption time series will be modelled as a 2D image, as will each of the 21 clusters in the model (one of these images will be the image representing the next day's consumption cluster). All this information will be needed for the supervised machine learning process carried out by the deep learning algorithms.

### Consumption centroid

<img width="657" alt="Captura de pantalla 2024-02-18 a las 14 02 55" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/88ecd86f-e984-4e7c-9d83-a5e93b323d59">

### Modelling time series images for deep learning model training

To obtain better results in training models based on time series images with deep learning techniques, the time series instead of being modelled as curves (shown above), can be modelled by applying a series of transformations that generate a 2D representation in a specific  domain.
 The proposed methodology will model each weekly consumption time series as an image in this specific domain. The same process will be carried out for the 21 centroids representing the next day's consumption. 
 
 Deep leaning models will be trained by providing as inputs the weekly images, and as outputs the images of the consumption cluster for next day, both in this specific format.
 
To make predictions using this model, the weekly time series to be predicted will be converted to this specific domain, and the output generated will be a daily image in this specific domain. Finally, the time series will be obtained by applying the inverse process to the transform and will be represented as a curve.

**Example of week consumption time serie in Gramian Field Domain** (The inputs of neural networks)

![diff_t0](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/84e77093-755f-4cf6-9f66-d624d3c974d8)

**Example of cluster consumption time serie in Gramian Field Domain** (The output of neural networks)

![clus_0_gram](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/2c8a1496-72fd-49b4-9d92-55c9c322b51b)

 ## Deep Learning with Time Series Imaging

Neural network technologies for image Generation models:

- Variation Autoencoders (VAEs)
- Generative Adversarial Models (GANs)
- Auto Regression Models
- Diffusion Models


## Authors ✒️

* **Francisco Rodríguez Gómez** 
* **José del Campo Ávila** 
* **Llanos Mora López** 

## License 📄

  This project is under GNU General Public License v3.0 [LICENSE.md](LICENSE.md) for more detail.

## Acknowledgements 🎁

  This work has been supported by the project RTI2018-095097-B-I00 at the 2018 call for I+D+i Project of the Ministerio de Ciencia, Innovación y Universidades, Spain. 
