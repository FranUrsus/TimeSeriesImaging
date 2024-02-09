import importlib
import matplotlib.pyplot as plt

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


#plot a time serie
def plotTimeSeries (timeSerie):
    timeSerie.plot()
    plt.xlabel('Hour', fontsize=15)
    plt.ylabel('Consumption (Wh)', fontsize=15)
    plt.show()



            
        

    