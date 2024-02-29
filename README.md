 #  Deep learning and time series imaging for the next day electricity consumption forecasts

<img width="1024" alt="Captura de pantalla 2024-02-22 a las 14 51 47" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/266a5ad7-345a-4975-b918-72e99e483238">

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

In the example that will be used to describe the methodology, a data set with 529 consecutive weekly consumptions of the same consumer has been used.

> **_NOTE:_** This script does not prepare the data as described.

## Clustering model

The centroids of the model that has been used to assign the next day's cluster to each weekly consumption are as follows:

<img width="727" alt="centroids" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/d6e7827a-af06-4bf9-9f90-20e59d5a4962">

This model is available for free in model folder.

## Time series imaging

Time series can be modeled as 2D curves representing the value for each instant in the time series, but this way of modelling time series is not the most suitable for training models with deep learning algorithms to make predictions from time series images.

### Week consumption

This figure shows a time series of a user's weekly consumption represented as a curve.

![this](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/76bcc5f3-e6df-49c7-b636-0c5b0e17c9ee)

### Consumption centroid

This figure shows a time series of a  consumption centroid represented as a curve.

<img width="657" alt="Captura de pantalla 2024-02-18 a las 14 02 55" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/88ecd86f-e984-4e7c-9d83-a5e93b323d59">


### Modelling time series images for deep learning model training

To obtain better results in training models based on time series images with deep learning techniques, the time series instead of being modelled as curves (shown above), can be modelled by applying a series of transformations that generate a 2D representation in a specific  domain.
**The proposed methodology will model each weekly consumption time series as an image in the Gramian Angular Field domain**. 

 Modelling a time series as a differential GRAM matrix allows to represent how different each value of the time series is with respect to the rest of the values of the series, as shown in the following example:
 
**Example of week consumption time serie in Gramian Field Domain**

![diff_t0](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/84e77093-755f-4cf6-9f66-d624d3c974d8)

This type of images will be used by deep learning algorithms to train models capable of predicting the next day's hourly consumption based on weekly consumption. 

### Deep learning

  Deep leaning model will be trained by providing as inputs the weekly temporal series modelled as Gramian Angular Field as RGB images. The neural netorks outputs will be the corresponding label for the consumption cluster for next day.

### Input

The following figure shows an example of some weekly consumptions modelled using GRAM matrices, in which the difference between each value of a time series with the rest is plotted for each of the weekly time series.

![test](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/01dec3a8-6ad4-42a1-a245-f599440ba393)



#### Output
The output of the network will be a one hot encoding vector that will have a value of 1 in the neuron that activates the predicted consumption cluster for the next day. The rest of the unselected clusters will remain at 0.

In this output example, the week consumption image with feed the deep learning model activates the first output neurons (cluster 0 next day consumption will be predicted)

**one_hot_encoding_output** = [1,0,0,0,0,0,0,0,0,...0]

<img width="386" alt="Captura de pantalla 2024-02-28 a las 12 17 19" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/9d092ac6-a684-4f3e-aa7e-db2a2ece4e15">

### Data preparation for deep learning model training

The Gramm matrix images of the weekly consumption time series will be stored in a folder with the name of the class they belong to (next day cluster).

next_day_cluster_0 .. next_day_cluster_n

<img width="553" alt="Captura de pantalla 2024-02-27 a las 16 45 08" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/02fd82e2-9907-4afb-b80e-327238b9a393">

<img width="599" alt="Captura de pantalla 2024-02-27 a las 16 50 44" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/0c62304c-c877-4479-8b69-cc162ff8394d">

**Split dataset in train and validation data**

Once the weekly consumption RGB images (Gram matrices) are available, and organized in folders according to the class to which they belong (next day cluster), the data set is divided into validation and test as a previous step to the training of deep learning models.

<img width="812" alt="Captura de pantalla 2024-02-28 a las 11 59 46" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/ae430f38-3f32-48a4-8065-f36c517b9b15">

### Deep learning model architecture


The proposed deep learning architecture consists of a series of neurons connected along a series of convolution and reduction layers (pooling layers) finally connected to a fully connected neural network.

The convolution operations allow the model to learn features from the images (the deeper the images, the more detail), and the reduction operations extract the relevant features and reduce the size of the neurons to speed up and enable the learning process.

For pooling, 2x2 kernels are used to halve the size of the neurons modeling feature maps.

Three convolutional blocks (32,64,128) are proposed for detailed learning of relevant features and patterns in the images.

All the output neurons that connect the kernels of the convolutional blocks with the inputs of the feature map neurons are connected to a relu. Each feature map goes through a 2x2 maxpooling block to halve its size.

Finally the n-dimensional model is flattened and passed through 21 neurons to which the softmax function is applied to obtain the probability that the next day of consumption of the input weekly consumption image belongs to each of the 21 centroids.


#### Predictions

To make predictions with the trained model, the weekly time series to be predicted would be first converted to Gramian domain. The output activated by the neural network will be the predicted next day consumption cluster label. 

## Authors ✒️

* **Francisco Rodríguez Gómez** 
* **José del Campo Ávila** 
* **Llanos Mora López** 

## License 📄

  This project is under GNU General Public License v3.0 [LICENSE.md](LICENSE.md) for more detail.

## Acknowledgements 🎁

  This work has been supported by the project RTI2018-095097-B-I00 at the 2018 call for I+D+i Project of the Ministerio de Ciencia, Innovación y Universidades, Spain. 
