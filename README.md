 #  Deep learning and time series imaging for the next day electricity consumption forecasts

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
  
  The first step consists of preparing a dataset in which each row represents the weekly hourly consumption (24x7) of any user and the cluster (cluster_ next_day) that best represents the next day's (24 hours) consumption for that user. This cluster is assigned by applying a clustering model to the 24 hours of consumption of the next day. This model will be available for free and will be detailed as below.

This figure shows what the data must look like in order to be processed by the script:

<img width="1008" alt="Captura de pantalla 2024-02-19 a las 19 03 33" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/c85737ec-c687-4f9b-bc32-c5b6d673bf31">

In the example that will be used to describe the methodology, a data set with 150724 consecutive weekly consumptions of multiple consumer has been used.

Due to the size of the dataset, the consecutive weekly consumptions of the first 10 consumers have been selected. (A total of 5223 weekly consumption rows has been obtained)

> **_NOTE:_** This script does not prepare the data as described.

## Clustering model

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
The output of the network will be a one hot encoding vector that will have a value of 1 in the neuron that activates the predicted consumption cluster for the next day. The rest of the unselected clusters will remain at 0.

In this output example, the week consumption image with feed the deep learning model activates the first output neurons (cluster 0 next day consumption will be predicted)

**one_hot_encoding_output** = [1,0,0,0,0,0,0,0,0,...0]

<img width="386" alt="Captura de pantalla 2024-02-28 a las 12 17 19" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/9d092ac6-a684-4f3e-aa7e-db2a2ece4e15">

### Data preparation for deep learning model training


**Split dataset in train and validation data**

To avoid overtraining and to be able to evaluate the performance of the deep learning models generated by the algorithms, the dataset has been divided into training and validation. 75% has been reserved for training (3917 weekly consumptions) and the remaining 25% (1306 weekly consumptions).

Next plot shows the number of images on each next_day consumption cluster:

![cluster_split](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/7589c90c-9152-4052-a1ac-acdfa7f296b4)


The Gramm Summation, Difference and Markov images channels for each weekly consumption time series will be stored in a folder with the name of the class they belong to (next day consumption cluster).

next_day_cluster_0 .. next_day_cluster_n

<img width="553" alt="Captura de pantalla 2024-02-27 a las 16 45 08" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/02fd82e2-9907-4afb-b80e-327238b9a393">

<img width="797" alt="Captura de pantalla 2024-03-14 a las 19 06 14" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/cefbb3d3-381b-4ec8-a4ef-31c21268d5cd">

<img width="832" alt="Captura de pantalla 2024-03-14 a las 19 06 40" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/0b6d9452-6c61-41f5-ac40-47c9170bebde">



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
