 #  Deep learning and time series imaging for next day electricity consumption forecasts

<img width="863" alt="Captura de pantalla 2024-05-09 a las 20 49 50" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/d258f9a2-7334-42f7-9113-1ba097dcf649">


<!---**Alternative 2**


-->

This script implements a methodology that combines ***deep learning***, ***and time series imaging*** to ***predict hourly electricity consumption for the next day***.

The proposed methodology consists of 3 phases:

- **Data preparation**.Data preparation for model training
- **Time series imaging**. Image generation from time series data
- **Training of Deep learning models:** Use deep learning algorithms to obtain models for next day hourly consumption forecasting using obtained imaged from time series .

## Data preparation 

- For each user, get the maximum number of consecutive days of consumption. (24 time values ​​on each row). Each row will contain the 24 consumption values ​​of the day following the previous row.

- In each row, add the 24 hourly consumption values ​​for the 23 consecutive days.

- Finally, add to the end of each row, the 24 hourly consumption values ​​of the 25th.

- In each row there will be 24x24 values ​​for the hourly consumption of 24 days, and the 24 consumption values ​​for the following day. (600 columns)


## Time series imaging

Time series can be modeled as 2D curves representing the value for each instant in the time series, but this way of modelling time series is not the most suitable for training models with deep learning algorithms to make predictions from time series images.

### Week consumption

This figure shows a time series of a user's weekly consumption represented as a curve.

![curve](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/0bbfee4a-c0ac-41f0-85af-a99da1a9755b)


### Consumption centroid

This figure shows a time series of a  consumption centroid represented as a curve.

<img width="387" alt="Captura de pantalla 2024-03-14 a las 15 50 08" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/19a32e90-89df-4b45-ae3a-5dcdfb44e4cc">



### Modelling time series images for deep learning model training

To obtain better results in training models based on time series images with deep learning techniques, the time series instead of being modelled as curves (shown above), can be modelled by applying a series of transformations that generate a 2D representation in a specific  domain.

**The proposed methodology models each 1D curve of each weekly time series as an three channel image in which the pixels of each channel are represented by the Gramian Angular Field Summation and difference and the Markov transition models. The information that each of these models provides to the time series will be of great help for training and pattern discovery in the neural network.**. 
 
**Example of week consumption time serie in Gramian Field Domain**

 Modelling a time series as a differential GRAM matrix allows to represent how different each value of the time series is with respect to the rest of the values of the series, as shown in the following example:

![249](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/93e7a5e2-dd9e-4349-b817-d966c440ff6a)


**Example of week consumption time serie in Markot Transition Domain**

A Markov transition matrix models the transition probabilities for a time series. You can see an example in the following figure.

![249_mk](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/a4f99004-66d0-4ed9-9336-edea70e148c8)

These images will be used by deep learning algorithms to train models capable of predicting the next day's hourly consumption based on weekly consumption. 

### Deep learning

  Deep leaning model will be trained by providing as inputs the three channels of images of weekly temporal series (Gramian Angular Field Summation, Gramian Angular difference, and Markov Transition). The neural netorks outputs will be the corresponding label for the consumption cluster for next day.

### Input

The input to the deep learning algorithms will be a 4-dimensional n-dimensional array data structure detailed in the following figure for the x-values and one hot encoding class laber for y-values.

<img width="708" alt="Captura de pantalla 2024-03-22 a las 13 50 14" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/cfeeee01-b733-426f-bd72-d20ddbc941f7">

#### Output


The output of the model will be a soft layer connected on each fully connected layer neurons to evaluate  the predicted next day consumption cluster. 

<img width="408" alt="Captura de pantalla 2024-03-15 a las 17 08 06" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/c2f17d24-ca28-4aba-8e4c-ed89a921d8b8">


### Data preparation for deep learning model training


**Split dataset in train and validation data**

Eighty percent of the dataset has been selected for training and validation of the models (75%  - 25%), and 20% has been reserved for model testing. 

***Train data:*** 6003 observations
***Validation data:*** 2001 observations
***Test data:*** 2001 observations


Next plot shows the number of images on each next_day consumption cluster:

![cluster_obs](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/db03eb8b-9e5b-415e-84b3-5788c26fc63d)

The following images show examples of two weekly time series of electricity consumption modelled as Gramian and Markov images and whose next day's consumption belongs to clusters 15 and 13.

<img width="621" alt="Captura de pantalla 2024-03-17 a las 11 31 20" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/eca7467f-0747-42e5-a8c5-cec57df46075">

<img width="624" alt="Captura de pantalla 2024-03-17 a las 11 35 01" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/b8f62bba-24a8-421a-826f-a370f7b780ef">

***Input***

At this point, we would already have the input data prepared with the images for each of the three channels corresponding to each weekly consumption for the deep learning algorithms.

***Train***

<img width="216" alt="Captura de pantalla 2024-03-22 a las 13 26 47" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/6996ba1e-2e2d-48e9-a70a-e650bcb98664">

***Validation***

<img width="211" alt="Captura de pantalla 2024-03-22 a las 13 28 08" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/43f0621d-cdc4-40a9-854a-1efbfb8e1491">


***Labels***

The first step consists of encoding each of the classes of each observation of the training and validation set with an array that will be assigned the value 1 in the cluster to which the weekly consumption belongs and zeros for the rest of the clusters.

***Label_train***

<img width="885" alt="Captura de pantalla 2024-03-22 a las 13 34 17" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/9ec7d8c2-5c2b-48af-9353-dc123ea580f4">


***Label_validation***

<img width="882" alt="Captura de pantalla 2024-03-22 a las 13 35 46" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/e04f9b2e-c655-4962-8df6-2a6f1f6e9c6a">








### Deep learning process

The proposed script implements a hypertuning search method for get the best model based on the evaluation of the results obtained from different models generated by experimenting with different deep learning architectures and different configurations of a series of parameters relevant in the training processes of deep learning models.

***(cambiar por una captura de todas las configuraciones probadas y de la configuración del algoritmo que ha generado el mejor modelo)***
<img width="1152" alt="Captura de pantalla 2024-03-01 a las 14 25 53" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/18eb6e3c-cb82-450a-9d17-1556359fd094">

### Deep learning model architecture

***(cambiar por la arquitectura del mejor modelo obtenido con hyperparameter tuning)***
![metodologia_DL_TS_forecasting](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/76c57d16-7202-4575-9f38-12ce379cdf58)

The deep learning model consists of a first phase in which a series of neurons (input image segmented into 3 channels) are connected to a number of convolution filters or kernels (neurons) through a series of convolution layers whose outputs will obtain as many feature maps as filters.  A series of pooling or reduction layers will be applied to these feature maps to reduce the size of the maps. Finally, the deep learning model connects a fully connected neural network with a softmax layer that will allow the predictions of the consumption cluster of the next day.

The convolution operations allow the model to learn features from the images (the deeper the images, the more detail), and the reduction operations extract the relevant features and reduce the size of the neurons to speed up and enable the learning process.

For pooling (reduction), 2x2 kernels are used to halve the size of the neurons modeling feature maps.

All the output neurons that connect the kernels of the convolutional blocks with the inputs of the feature map neurons are connected to a relu function block.

Finally the n-dimensional model is flattened and passed through 21 neurons to which the softmax function is applied to obtain the probability that the next day of consumption of the input weekly consumption image belongs to each of the 21 centroids.

#### Predictions

To make predictions with the trained model, the weekly time series to be predicted would be first converted to Gramian domain. The output activated by the neural network will be the predicted next day consumption cluster label. 


## LSTM 

En este enfoque se van a entrenar modelos basados en Deep Learning (LSTM) predecir series temporales diarias a partir de las series temporales de consumo semanal sin convertir las series temporales en imágenes.
El primer paso consiste en preparar el conjunto de datos de forma adecuada para el entrenamiento de modelos con redes neuronales LSTM.

<img width="854" alt="Captura de pantalla 2024-03-06 a las 14 09 46" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/1a5044cf-4e75-4dcc-9d53-acf8eec11b4d">

### LSTM Models


## Authors ✒️

* **Francisco Rodríguez Gómez** 
* **José del Campo Ávila** 
* **Llanos Mora López** 

## License 📄

  This project is under GNU General Public License v3.0 [LICENSE.md](LICENSE.md) for more detail.

## Acknowledgements 🎁

  This work has been supported by the project RTI2018-095097-B-I00 at the 2018 call for I+D+i Project of the Ministerio de Ciencia, Innovación y Universidades, Spain. 
