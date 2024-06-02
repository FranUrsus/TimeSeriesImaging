 #  Deep learning and time series imaging for next day electricity consumption forecasts

 <img width="1461" alt="Captura de pantalla 2024-06-02 a las 16 36 57" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/ee1db344-5346-49c9-8371-905a8e6092d1">


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

- Split consumers into train, validation and test.

- Reserve for the test set (on test consumers) days later in time than the days used for training and validation.

   - For training and validation of users who have fallen into these sets, consecutive consumption data from one year have been used.
     
   - For the users of the test set, the consumption of days longer than one year has been used to evaluate the predictive quality of the models with a real future case.

- Each day of the month value is modeled cyclically as follows
  
<img width="457" alt="Captura de pantalla 2024-05-26 a las 16 30 57" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/a7411a06-3b07-42fa-80d6-12467066c373">

- The same methodology has been applied for each day of year x [0-181] ->X, x[182 || 183 ||184]->182, x[x>184] = abs(365-x)+1 



## Time series imaging

The proposed methodology generates 3 different images from each row of the dataset described above.

- The first image (consumption image) (24x24) represents the hourly consumption for each of the 24 days

- The second image (day of year image) (24x24) model the information about the day of year for each hour of consumption 

- The third image models the day (day of the week) (24x24) of the week for each consumprion hour


<img width="776" alt="Captura de pantalla 2024-05-11 a las 21 55 44" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/870105aa-10a4-4417-a6e0-b6c584f4b873">


**Data normalization** 

The hourly values ​​for each of the 3 images have been normalized between 0 and 1. For the consumption image, they have been normalized between the highest and lowest consumption of all users. For the images of day of the year and day of the week  


### Deep learning. (Modeling)

A custom deep learning architecture will be created from scratch. This backbone will be the basis for training each of the models (24) to make predictions for each of the hours of the following day.

<img width="955" alt="Captura de pantalla 2024-05-09 a las 21 28 03" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/af47c2aa-67ce-4816-b8ff-7cb95b13e418">


**X-values** = 24 daysx 24hours (3 channels images)

**y_true_values** = consumption_h_hour for next day

<img width="863" alt="Captura de pantalla 2024-05-09 a las 20 49 50" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/378d229a-4ab0-400d-84a7-773e82e1788c">

### Error models (24)
![plot1](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/e3b1f58e-cf49-4f93-8701-6b72cfd4f375)

![plot2](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/7e572eb3-6d52-45af-a500-5a8045651827)

![plot3](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/55b05c15-a284-4a7f-9a6a-f42a1c6b5314)

![plot4](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/0bec953b-14c7-4321-9c99-fe1098ea8866)

### Model evaluation (Baseline)

To evaluate the quality of the 24 models trained on one year's consumption data, predictions have been made on the test set data with the 24 proposed models and 24 persistent models (predicted value for an hour is the value of that hour for the previous day).

![MAES](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/06176781-6893-4876-8ee9-8f378bbcd4ee)
![rMAES](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/cab8a2de-c00c-4dc3-9e6b-174bccad81a3)
![RMSES](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/13c84849-5477-4fce-89ba-495b0409973a)
![rRMSES](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/01d28fb3-154d-4b83-ae84-31aa0ce25895)
![R^2](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/9b30a0d7-4829-4845-961a-7bee0a32090c)
![score](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/403397f8-ffec-45d2-a320-04a876143da9)



## Authors ✒️

* **Francisco Rodríguez Gómez** 
* **José del Campo Ávila** 
* **Llanos Mora López**
* **Ezequiel López Rubio**

## License 📄

  This project is under GNU General Public License v3.0 [LICENSE.md](LICENSE.md) for more detail.

## Acknowledgements 🎁

  This work has been supported by the project RTI2018-095097-B-I00 at the 2018 call for I+D+i Project of the Ministerio de Ciencia, Innovación y Universidades, Spain. 
