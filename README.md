 #  Deep learning, LSTM and time series imaging for the next day electricity consumption forecasts

![ESQUEMA_TS](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/21d83c21-d242-4bbe-9aa1-fc5a7a7ead97)


<!---**Alternative 2**
<img width="956" alt="Captura de pantalla 2024-02-19 a las 19 31 14" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/4c39b5b4-267b-4664-928d-8adf0a139f31">
-->

This script implements a methodology that combines ***deep learning***, ***and time series imaging*** to ***predict hourly electricity consumption for the next day***.

The proposed methodology consists of 3 phases:

- **Data preparation**. Prepare the data accordingly to apply the methodology
- **Time series imaging**. Transformation of time series into 2D images suitable for training models with deep learning. 
- **Training of Deep learning models:** Use deep learning algorithms to obtain models capable of predicting the next day's consumption from 2D images of weekly time series.

## Data preparation 

### Deep Learning with imaging time series
  
  The first step consists of preparing a dataset in which each row represents the weekly hourly consumption (24x7) of any user and the cluster (cluster_ next_day) that best represents the next day's (24 hours) consumption for that user. This cluster is assigned by applying a clustering model to the 24 hours of consumption of the next day. This model will be available for free and will be detailed as below in clustering model centroids section.

This figure shows what the data must look like in order to be processed by the script:

<img width="1008" alt="Captura de pantalla 2024-02-19 a las 19 03 33" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/c85737ec-c687-4f9b-bc32-c5b6d673bf31">

In the example that will be used to describe the methodology, a data set with 150724 consecutive weekly consumptions of multiple consumer has been used.

For the methodology based on deep learning with time series images, weekly consumption observations assigned to very infrequent clusters have been eliminated. In this way, the aim is to reduce the error and improve the learning quality of the models. The data set has been balanced so that there is an adequate minimum number of observations in the different consumption clusters for the next day.

After balancing, the weekly consumption assigned to clusters 20,4,1,7,18,2 has been eliminated, due to the low frequency. Finally, the possible consumption clusters for the next day are: [0,3,5,6,8,9,10,11,12,13,14,15,16,17,19]

<img width="383" alt="Captura de pantalla 2024-03-17 a las 10 34 28" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/51055cd1-ee01-4e3e-ad65-3ece8ec2f024">
<img width="474" alt="Captura de pantalla 2024-03-17 a las 10 34 22" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/f26ac5e4-5161-4ce7-8d3c-4e8c5aeaa2c1">

This cut-off value has been used after analyzing the data set and seeing that below that value there are very few weekly consumption observations.

<img width="389" alt="Captura de pantalla 2024-03-17 a las 10 23 17" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/df441020-4ddd-46b6-bb81-099899fd92b6">

After balancing, the total data set is left with **10,005 weekly consumption observations** spread equally across 15 consumption clusters for the next day.

### Deep Learning with LSTM

To train deep learning models with LSTM neural networks, the data has been prepared in such a way that after each weekly consumption row (168 consumption values) the daily consumption (24 values) of the next day's consumption is attached. In this way, the rows represent consecutive weekly consumptions of different users and the next day's consumption for each of these weekly consumptions.

<img width="809" alt="Captura de pantalla 2024-03-17 a las 10 57 07" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/a8ce4644-f8f4-4e23-8d7f-e9de8b88da65">



> **_NOTE:_** This script does not prepare the data as described.

## Clustering model centroids

The centroids of the model that has been used to assign the next day's cluster to each weekly consumption are as follows:

<img width="727" alt="centroids" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/d6e7827a-af06-4bf9-9f90-20e59d5a4962">

This model is available for free in model folder.

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

A sequence of week consumption (three channels images) - explained in time series imaging section-


![249_mk](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/387e570f-b61b-4d6f-a467-684d85b17a46)

![249](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/edc652c8-7e09-4a97-94db-89068cc900d5)


#### Output

The output of the model will be a soft layer connected on each fully connected layer neurons to evaluate  the predicted next day consumption cluster. 

<img width="408" alt="Captura de pantalla 2024-03-15 a las 17 08 06" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/c2f17d24-ca28-4aba-8e4c-ed89a921d8b8">


### Data preparation for deep learning model training


**Split dataset in train and validation data**

To avoid overtraining and to be able to evaluate the performance of the deep learning models generated by the algorithms, the dataset has been divided into training and validation. 75% has been reserved for training (7503 weekly consumptions) and the remaining 25% (2502 weekly consumptions).

<img width="969" alt="Captura de pantalla 2024-03-15 a las 18 29 46" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/e3f8fbbb-2cf7-42a1-9208-efe0ce12bee0">


Next plot shows the number of images on each next_day consumption cluster:

![cluster_obs](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/db03eb8b-9e5b-415e-84b3-5788c26fc63d)

The following images show examples of two weekly time series of electricity consumption modelled as Gramian and Markov images and whose next day's consumption belongs to clusters 15 and 13.

<img width="621" alt="Captura de pantalla 2024-03-17 a las 11 31 20" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/eca7467f-0747-42e5-a8c5-cec57df46075">

<img width="624" alt="Captura de pantalla 2024-03-17 a las 11 35 01" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/b8f62bba-24a8-421a-826f-a370f7b780ef">



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
