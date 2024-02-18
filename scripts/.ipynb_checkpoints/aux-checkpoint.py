import importlib
import matplotlib.pyplot as plt
import random

# return if a library is installed
def checkLibraryInstalled(libraryName):
    print(libraryName)
    installed = True
    try:
        importlib.import_module (libraryName)
        print(libraryName + " already installed")
    except ImportError as e:
        print("Error -> ", e)
        installed = False
    return installed

# return a generated random color
def random_color_generator():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)
 

#plot a time serie with week separator
def plotTimeSeriesWithWeekSeparator (timeSerie):
    plt.figure(figsize=(20,10))
    timeSerie.plot(marker='o', markersize=3)
    plt.xlabel('Hour', fontsize=15)
    plt.ylabel('Consumption (Wh)', fontsize=15)
    weekSeparatorVerticalLineCoords =  [*range(23, 169, 24)] 
    i = 0
    colors = ['blue','green','red','cyan','pink','yellow','orange']
    for weekCoords in weekSeparatorVerticalLineCoords:
        plt.vlines(x=weekCoords,colors=colors[i], ls='--', lw=2, label='day '+str(i+1), ymin = 0, ymax = max (timeSerie) )
        i+=1
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', title="Day of the week")
    plt.show()

#plot a time serie
def plotTimeSeries (timeSerie):
    timeSerie.plot()
    plt.xlabel('Hour', fontsize=15)
    plt.ylabel('Consumption (Wh)', fontsize=15)
    plt.show()

            
        

    