 # Prediction of electrical energy consumption based on Time Series Imaging

<img width="800" alt="Graphical_Abstract" src="https://github.com/ursusdm/URSUS_UHI/assets/68539118/2d23532d-7795-41a7-ab77-0d5f3089e60a">

This script implements a methodology that combines ***deep learning***, ***image and time series processing*** to ***predict hourly electricity consumption for the next day***.

The tool is very simple to use and consists of the following steps.

  ## Data preparation
  
  The first step consists of preparing a dataset in which each row represents the weekly hourly consumption (24x7) of any user and the cluster (cluster_ next_day) that best represents the next day's (24 hours) consumption for that user. This cluster is assigned by applying a clustering model to the 24 hours of consumption of the next day that will assign the day of consumption to one of 21 clusters. This model will be available for free.

<img width="993" alt="Captura de pantalla 2024-02-09 a las 17 04 01" src="https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/a6a96b4c-3cb8-401a-b068-80e1ed09d749">

> **_NOTE:_** This script does not prepare the dataset as described, but the clustering model of the day's hourly consumption with the 21 centroids shown below is offered for free in this repository, in case you need to use it to generate the next day's cluster column. You can use any model of your own.

### Clustering model

![fig--k21cent-Ireland](https://github.com/FranUrsus/TimeSeriesImaging/assets/68539118/9cb4ceef-1f5d-44c6-90a5-0ab72116627e)

   
  3. **Time Series Imaging** 

      In this phase, each row (the 24x7) first columns will be converted into an image using the technique...
  
      2.2. **Load** the image and **crop the area of ​​interest** interactively on the map

      2.3. **Show results**

   4. **Deep Learning with Time Series Imaging** 






## Download a Landsat-8 image of the city of interest

To download Landsat-8 images of the cities of interest for the study, we can do so free of charge from the site  https://earthexplorer.usgs.gov . It is necessary to be registered on this site in order to download the images.

The following screenshots show the process in detail: 

### Select the city of interest for the study (Search Criteria tab)

Define a criterion to filter the city of interest. There are different alternatives, for example:

  1. If it is the capital of a country or first-order administrative division. The "Feature (GNIS)" geocoding method can be used. Here, the "Feature Name" and "Country" are defined (or "State" if US features are used). For example, if we are interested in Madrid, the capital of Spain we will get
   <img width="300" alt="Select the city of study interest (1): Madrid" src="https://github.com/ursusdm/URSUS_UHI/assets/68539118/88bfab5f-0eaa-48df-9270-beee435b553d">
 
 2. If it is not listed with the "Feature (GNIS) geocoding method", the "Address/place" can be used. For example, if we are interested in Malaga, the capital of one Spanish province we will get
   <img width="300" alt="Select the city of study interest (1): Malaga" src="https://github.com/ursusdm/URSUS_UHI/assets/33939770/e10500b0-d082-41bb-8d0a-b80d5156d0c0">


Then, select the most appropriate element and press the "Show" button. A map with the selected item is then displayed.

The rest of the examples in this manual will use Malaga as the city of interest.

<img width="800" alt="Select the city of study interest (2): Map" src="https://github.com/ursusdm/URSUS_UHI/assets/33939770/f8c45581-2be9-4093-8d17-e60883e8c0e1">

### Select satellite image type: LANDSAT-8 (Data Sets tab)

Search for Landsat-8 image and press the "Results" button

<img width="300" alt="result_button" src="https://github.com/ursusdm/URSUS_UHI/assets/68539118/ec40f2e3-c1c7-433f-b61a-7178a4627cbe">


### Select the image that most suits your needs and download it

Each image has a list of buttons to interact with it. Click on the "Download" button for that image (green arrow going down to a hard disk).

<img width="700" alt="Select the image that most suits" src="https://github.com/ursusdm/URSUS_UHI/assets/33939770/21211a9e-5221-4736-96d0-18ec701c9786">

Click in the "Products options" button

 <img width="300" alt="Products options button" src="https://github.com/ursusdm/URSUS_UHI/assets/68539118/26319dfb-edca-42f5-808f-3741c6a1daed">

And download the first file (1 GB aprox.). It may be necessary to allow pop-up windows

  <img width="500" alt="Download file" src="https://github.com/ursusdm/URSUS_UHI/assets/33939770/ef13bda8-2214-4eb1-a68d-37991ecea9d6">

After this process, the data of the city of interest is ready to be processed by the URSUS-UHI software.

> [!NOTE]  
> All downloaded images must be placed unzipped in the data folder of the project.

## Run URSUS_UHI software to process the image to detect unfavourable areas

### Requirements

- R 4.3.1 or latest version
- RStudio
- Some R libraries (renv, shiny, dplyr, ...)

### Install / Run URSUS_UHI software
  
- **Clone the repository.** Open your terminal on your local machine and clone the repository

      git clone "https://github.com/ursusdm/URSUS_UHI.git"

- **Open R project file** placed in the folder
  
  <img width="121" alt="R file" src="https://github.com/ursusdm/URSUS_UHI/assets/68539118/27ed6bd7-fa32-4dee-8b95-09f2b65b5958">

- **Install / use renv package**. This package helps you to create reproducible environments for your R projects.

  Use the terminal inside RStudio to install the renv package.

      install.packages("renv")
 
  Download dependencies and libraries using renv:  

      renv::init()

  Select yes (if this is the first time renv is used) and select "Selection 1" to use exactly the same libraries that the application uses in its RStudio environment project.

  <img width="500" alt="renv library" src="https://github.com/ursusdm/URSUS_UHI/assets/68539118/fd47d9ee-1338-419e-bdff-cd197c44e1d5">

  If a package is not installed, you can try to install it manually.
  
      install.packages("libraryName")

  The required library packages are listed in the app.R file where there is code to automatically load such libraries.

  
## Load the image and crop the area of ​​interest interactively on the map 🚀
 
### Run the application

  Click button on "Run App" on file App.R
  
  <img width="400" alt="Run App in Rstudio" src="https://github.com/ursusdm/URSUS_UHI/assets/68539118/f2826ee4-3d62-4a81-adbe-642c0c0668e9">

  The app is executed in a new window

  <img width="300" alt="Running App" src="https://github.com/ursusdm/URSUS_UHI/assets/33939770/32637a0f-deb4-4ebf-93f3-cf417edf1226">

### Select the image

  Search for the folder with the desired images.

  <img width="150" alt="Select folder" src="https://github.com/ursusdm/URSUS_UHI/assets/33939770/41882cd5-dc82-4f9e-84d2-8006fcdce9d7">
  
  Select the folder (not any file inside). In this example, select Malaga folder and click "Select" button.

  <img width="500" alt="Select Malaga folder" src="https://github.com/ursusdm/URSUS_UHI/assets/33939770/64526344-1a82-4cae-bc77-438c0b647f46">


### Crop the area of interest over the map
The red rectangle, sometimes visible, is the area covered by the selected landsat-8 image, a very large area.

Move on the map, zooming in and out, and find the area of interest.

Use the icon with a black square to define this area. It is marked in the map by a blue rectangle. 

<img width="500" alt="Select area of interest" src="https://github.com/ursusdm/URSUS_UHI/assets/33939770/f1194338-4ee0-4997-a644-468b6247d97b">

### Process the urban area

  Next screenshots show the results calculated from the data.

 **Disadvantaged Area Index (DAI)** for the unfavourable area cluster
 
  <img width="600" alt="DAI map" src="https://github.com/ursusdm/URSUS_UHI/assets/33939770/713e4180-d817-4f10-9805-54aaaef9d683">

 **Additional map:** **LST** (Land Surface Temperature) map, **NDVI** (Normalized difference Vegetation Index) map and **Clusters** map (urban areas classified on different levels (favourable (fav.), more favourable (more fav.), unfavourable (unf.))
 
 <img width="300" alt="LST" src="https://github.com/ursusdm/URSUS_UHI/assets/33939770/47fb4b39-a078-4939-8cc4-8c18b6a371df">
 <img width="300" alt="NDVI" src="https://github.com/ursusdm/URSUS_UHI/assets/33939770/0dc04521-114d-40c8-91d4-a0abccf18a47">
 <img width="300" alt="CLUSTER" src="https://github.com/ursusdm/URSUS_UHI/assets/33939770/994a2e89-4622-4594-8134-961fd57a36a5">

 

## Authors ✒️

* **Francisco Rodríguez Gómez** 
* **José del Campo Ávila** 
* **Domingo López Rodríguez** 
* **Luis Pérez Urrestarazu** 

 
## License 📄

  This project is under GNU General Public License v3.0 [LICENSE.md](LICENSE.md) for more detail.

## Acknowledgements 🎁

  This work has been supported by the project RTI2018-095097-B-I00 at the 2018 call for I+D+i Project of the Ministerio de Ciencia, Innovación y Universidades, Spain. 
