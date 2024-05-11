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

- Split consumers into train, validation and test.



## Time series imaging

The proposed methodology generates 3 different images from each row of the dataset described above.

- The first image (consumption image) (24x24) represents the hourly consumption for each of the 24 days

- The second image (months image) (24x24) reflects the information of the month to which each hour of consumption belongs

- The third image models the day (day of the week) (24x24) of the week to which each hour of consumption belongs


<img width="776" alt="Captura de pantalla 2024-05-11 a las 21 55 44" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/870105aa-10a4-4417-a6e0-b6c584f4b873">


**Data normalization** 

The hourly values ​​for each of the 3 images have been normalized between 0 and 1. For the consumption image, they have been normalized between the highest and lowest consumption of all users. For the images of day of the month and day of the week associated with each consumption schedule, it has been divided by 10.


### Deep learning. (Modeling)

A custom deep learning architecture will be created from scratch. This backbone will be the basis for training each of the models (24) to make predictions for each of the hours of the following day.

<img width="955" alt="Captura de pantalla 2024-05-09 a las 21 28 03" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/af47c2aa-67ce-4816-b8ff-7cb95b13e418">


**X-values** = 24 daysx 24hours (3 channels images)

**y_true_values** = consumption_h_hour for next day

<img width="863" alt="Captura de pantalla 2024-05-09 a las 20 49 50" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/378d229a-4ab0-400d-84a7-773e82e1788c">


## Authors ✒️

* **Francisco Rodríguez Gómez** 
* **José del Campo Ávila** 
* **Llanos Mora López**
* **Ezequiel López Rubio**

## License 📄

  This project is under GNU General Public License v3.0 [LICENSE.md](LICENSE.md) for more detail.

## Acknowledgements 🎁

  This work has been supported by the project RTI2018-095097-B-I00 at the 2018 call for I+D+i Project of the Ministerio de Ciencia, Innovación y Universidades, Spain. 
