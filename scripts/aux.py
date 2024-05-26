import matplotlib.pyplot as plt
import random
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import datetime
from sklearn.cluster import KMeans
from tensorflow.python.client import device_lib
import tensorflow as tf


# return a generated random color
def random_color_generator():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r, g, b


# plot a time_series with week separator
def plot_time_series_with_week_separator(time_series):
    plt.figure(figsize=(20, 10))
    time_series.plot(marker='o', markersize=3)
    plt.xlabel('Hour', fontsize=15)
    plt.ylabel('Consumption (Wh)', fontsize=15)
    week_separator_vertical_line_coords = [*range(23, 169, 24)]
    i = 0
    colors = ['blue', 'green', 'red', 'cyan', 'pink', 'yellow', 'orange']
    for weekCoords in week_separator_vertical_line_coords:
        plt.vlines(x=weekCoords, colors=colors[i], ls='--', lw=2, label='day ' + str(i + 1), ymin=0,
                   ymax=max(time_series))
        i += 1
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', title="Day of the week")
    plt.show()


# plot a time_series
def plot_time_series(time_series):
    time_series.plot()
    plt.xlabel('Hour', fontsize=15)
    plt.ylabel('Consumption (Wh)', fontsize=15)
    plt.show()


def create_folder_structure(next_day_clusters):
    path = "../data/images/input/"
    for clus in next_day_clusters:
        path_clus = path + 'next_day_cluster_' + str(clus)
        if not (os.path.exists(path_clus)):
            os.mkdir(path_clus)


# Create folder structure over 1D folder to save original images classified by cluster next day
def create_folder_structure_origin_images(next_day_clusters):
    path = "../data/images/1D/"
    for clus in next_day_clusters:
        path_clus = path + 'next_day_cluster_' + str(clus)
        if not (os.path.exists(path_clus)):
            os.mkdir(path_clus)


# return list with class names
def get_cluster_class_names(next_day_clusters):
    classes = []
    for clus in next_day_clusters:
        path_clus = 'next_day_cluster_' + str(clus)
        classes.append(path_clus)
    return classes


def prepare_data_for_lstm(df):
    transposed_df = df.T
    df_unstack = transposed_df.unstack().reset_index(drop=True).T
    return df_unstack


# plot predicted consumption hourly day with lstm vs real consumption day
def plot_predicted_over_real(n_iter, prediction_no_scaled, y_test):
    for i in range(n_iter):
        plt.figure()
        plt.plot(prediction_no_scaled[i], color='red', label='predicted')
        plt.plot(y_test[i], color='green', label='real')
        plt.legend()
        plt.show()


# Split data in validation, train and test sets
def split_data(x, target):
    X_train, X_test, y_train, y_test = train_test_split(x, target, test_size=0.1, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=1)
    return X_train, X_val, y_train, y_val, X_test, y_test


def create_three_channel_dataset(df,
                                 g_dif_channel,
                                 g_sum_channel,
                                 g_markov_channel):
    g_df_images = np.zeros((len(df), 168, 168))
    g_sum_images = np.zeros((len(df), 168, 168))
    mk_images = np.zeros((len(df), 168, 168))
    counter = 0
    for g_diff, g_summ, mk in zip(g_dif_channel, g_sum_channel, g_markov_channel):
        g_df_images[counter] = g_diff
        g_sum_images[counter] = g_summ
        mk_images[counter] = mk
        counter = counter + 1
    return g_df_images, g_sum_images, mk_images


# Return df with obs on next_day_cluster array
def get_dataset_with_obs_in_clusters(cluster_array, data):
    df_filtered = data[data.next_day_cluster.isin(cluster_array)]
    return df_filtered


# Return df with obs on next_day_cluster array
def get_dataset_with_distinct_obs_in_clusters(cluster_to_remove, data):
    df_filtered = data[~data.next_day_cluster.isin(cluster_to_remove)]
    return df_filtered


# Sample the dataset with a number of observations for each cluster equal to the number of observations
# in the cluster with the fewest observations.

def sample_equitative_cluster_df(data):
    df = data.groupby("next_day_cluster", group_keys=False)
    df_balanced = pd.DataFrame(df.apply(lambda x: x.sample(df.size().min()))).reset_index(drop=True)
    return df_balanced


# Return dataset  for  top n user
def filter_dataset_by_top_n_user(data, n):
    user_array = data.consumer_id.unique()
    data_top_n_user = data[data.consumer_id.isin(user_array[0:n])]
    return data_top_n_user


# Remove Tomek Links for remove week consumptions ambiguous to class prediction
# and over sampling obs in minority class using SMOTE
def resample_tomek_links(data):
    X = data.drop('next_day_cluster', axis=1)
    y = data['next_day_cluster']
    resample = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
    X, y = resample.fit_resample(X, y)
    return X, y


# Remove Tomek Links for remove week consumptions ambiguous to class prediction
def remove_tomek_links(data):
    from imblearn.under_sampling import TomekLinks
    undersample = TomekLinks()
    X, y = undersample.fit_resample(data.iloc[:, 0:169], data.next_day_cluster)
    return X, y


def get_month_for_consumption_day(x, origin_data):
    d = origin_data + datetime.timedelta(days=x)
    return d.month


def get_day_of_week_for_consumption_day(x, origin_data):
    d = origin_data + datetime.timedelta(days=x)
    return d.weekday()


# get 21 centroids for kmean models
def load_centroids():
    centroids = [[666.40479893, 484.33481247, 403.49585082, 370.5242449,
                  363.99568771, 391.12255842, 496.10640185, 659.05330929,
                  732.29261459, 661.01002998, 626.41878595, 557.36233516,
                  555.74481743, 628.43159322, 623.64899298, 647.05002927,
                  688.73262013, 989.96810246, 1397.61470245, 1526.12345439,
                  1467.50739569, 1424.30494595, 1277.73625541, 1007.61093758],
                 [7988.88639466, 7268.74730775, 6804.77445776, 6640.17503413,
                  6434.40785682, 6337.86303655, 6328.51327165, 6650.20203246,
                  7212.94342484, 7726.6173214, 8306.49582891, 8685.71105718,
                  9029.67844684, 8995.13635674, 9000.76566055, 8950.58956469,
                  9127.54664038, 9296.29713332, 9482.04307599, 9707.37919005,
                  9684.54300015, 9512.9323525, 8907.89276505, 8432.03579554],
                 [5063.53569443, 4365.20704993, 3896.86967641, 3597.82525297,
                  3459.25408651, 3456.88007339, 3567.78494385, 3849.55665518,
                  4187.1664628, 4334.57300122, 4484.80946292, 4604.65072834,
                  4891.90164572, 4842.72334038, 4875.54475703, 5077.23446014,
                  5581.56388302, 6209.94340042, 6724.0742244, 7015.64422328,
                  7026.27376849, 6932.77082175, 6616.3797954, 5979.58968086],
                 [664.64971722, 484.24338371, 403.54985479, 369.60886776,
                  362.38005281, 393.0481233, 495.5519526, 690.89894635,
                  841.49293436, 824.32571513, 840.90577216, 814.3801635,
                  856.04135019, 902.06588095, 901.49747788, 1392.20289267,
                  2653.60199451, 2977.45762989, 1610.71474955, 1378.71938267,
                  1351.9737931, 1336.13024388, 1202.78810991, 964.67879676],
                 [8317.06892543, 8494.768477, 8141.3019432, 7593.31971433,
                  6861.97176549, 6048.78790899, 5365.8445441, 4482.35459226,
                  2397.45523999, 2033.88739412, 2330.18285999, 2274.50357084,
                  2217.54658695, 2044.82710513, 2013.83341638, 1976.31772131,
                  1890.80468361, 1553.51934895, 1301.88290982, 1304.58379007,
                  1332.65819631, 1404.31954825, 1763.12771965, 5746.45308088],
                 [320.96178046, 255.8529408, 229.07278026, 219.64003234,
                  219.84097373, 230.34301883, 278.94034518, 365.18098082,
                  400.59031706, 395.79920081, 372.54551802, 340.40825456,
                  343.47485796, 352.78843204, 339.48136236, 350.84684382,
                  383.45530162, 445.23904749, 459.05602476, 460.7941184,
                  481.0623598, 508.57178647, 488.21405724, 411.43217114],
                 [744.03409614, 531.72816415, 450.16188436, 419.39059399,
                  413.72691544, 434.3549074, 511.50830058, 770.96687654,
                  2036.33027511, 4275.53008057, 2376.44072609, 1269.39402463,
                  1136.45864168, 1099.6225371, 1008.03069178, 1040.88979876,
                  1179.39464241, 1439.46789521, 1541.92177868, 1564.42676691,
                  1509.7273492, 1470.47353409, 1314.84056047, 1037.55479173],
                 [1601.16603784, 1170.65591189, 959.14439882, 878.8354325,
                  843.33421184, 861.24551961, 1003.8459191, 1423.54288964,
                  2003.28488598, 2671.09454586, 3392.50585363, 3846.09160517,
                  4295.97386673, 4593.98673917, 4619.61335516, 4631.94401598,
                  4846.79961716, 5423.64528658, 5400.69708151, 4917.79620485,
                  4308.65682739, 3766.67399989, 3124.9490096, 2392.10758475],
                 [3143.24018488, 2796.65769111, 2539.80058164, 2408.06945887,
                  2377.95899979, 2403.00059722, 2524.84233486, 2824.19295285,
                  2941.967984, 2819.09906003, 2755.33350644, 2713.53733901,
                  2693.8324678, 2662.73995118, 2571.32922206, 2558.56930307,
                  2656.32319796, 2854.96917844, 2976.30460636, 2999.86770357,
                  3070.49085999, 3190.84841088, 3166.72546219, 3052.21214167],
                 [924.00874251, 661.89334664, 548.39566201, 500.16010645,
                  487.6160346, 507.34621424, 590.92546906, 818.70312708,
                  1197.05776447, 1920.45172322, 4071.36357951, 4294.43602129,
                  2469.61630073, 1647.86794411, 1424.84790419, 1408.84958084,
                  1557.61201597, 1876.66726547, 1996.53409182, 2021.52737192,
                  1928.34179641, 1826.09276114, 1607.99800399, 1261.05015303],
                 [812.21019229, 574.68051668, 472.65269536, 433.00153241,
                  427.64155564, 469.31139442, 633.07503918, 959.77076803,
                  1098.21715461, 991.33350593, 982.19459528, 955.48468988,
                  992.90856611, 1049.9502701, 969.16469576, 1005.36742891,
                  1312.31778997, 2916.27394481, 4615.82014107, 3177.27366491,
                  2249.42958604, 1897.53034035, 1592.4525932, 1210.68650638],
                 [642.57734751, 483.99367229, 403.49586352, 370.68959448,
                  359.51990161, 370.07461016, 416.89810214, 523.60774039,
                  728.8428407, 929.33392458, 1341.47870377, 1737.51633606,
                  2046.35477174, 1563.61767427, 1046.05980638, 909.43473924,
                  878.26403538, 957.53414122, 980.66428813, 948.69999323,
                  956.18663598, 993.69464491, 951.83177397, 803.69112901],
                 [878.4607494, 650.14644639, 533.69427373, 472.9124521,
                  452.51305039, 463.77182597, 528.63582713, 700.44790688,
                  930.56599306, 1070.47802943, 1222.95578772, 1297.33524329,
                  1498.10612031, 2541.81275757, 3764.96807895, 3080.80275287,
                  1989.60919312, 1668.09438038, 1638.56510737, 1642.48099378,
                  1590.33541501, 1571.93889632, 1436.67476141, 1183.35150206],
                 [1214.39438737, 1128.04979955, 1082.22638436, 1067.5663994,
                  1058.08312453, 1050.55712854, 1090.87214984, 1445.62189927,
                  3157.46172638, 7300.26265347, 8970.56746429, 9258.00745427,
                  9114.90967176, 8316.58926334, 8724.38893761, 8678.63148334,
                  8195.13248559, 6048.81589827, 2487.61644951, 1572.3070659,
                  1318.12722375, 1259.56840391, 1214.6663117, 1195.21485843],
                 [3727.43318769, 3534.45903878, 3398.14127071, 3392.89249954,
                  3368.24003277, 3374.38494447, 3544.37156381, 4360.42799927,
                  6742.08747497, 9087.88376115, 9860.98124886, 9872.27571455,
                  9851.59894411, 9491.13890406, 9352.42508647, 9260.12980157,
                  9077.31139632, 7964.41307118, 6086.85882032, 5149.9208083,
                  4354.26697615, 3901.54951757, 3563.21017659, 3571.9576734],
                 [913.61542846, 781.63053205, 724.87335784, 706.45008692,
                  704.80832692, 716.9262489, 825.49348836, 1145.99270468,
                  2275.2919837, 4475.50676812, 5440.60058135, 5554.70519506,
                  5347.03126158, 4596.07266821, 4666.11293494, 4441.23652788,
                  4042.72155253, 2942.87446924, 1637.88943034, 1323.41953777,
                  1210.021373, 1164.52400901, 1074.35912342, 965.97936793],
                 [1104.78540487, 744.46498413, 585.5849443, 523.93598681,
                  506.04348665, 541.99653327, 676.98950022, 969.2010145,
                  1072.42832514, 945.66657746, 950.52198917, 937.85971868,
                  967.88452107, 1014.9711396, 960.02245597, 1022.0998444,
                  1160.32115516, 1427.85995519, 1800.21212423, 2745.17020601,
                  3488.90814713, 3503.88833634, 2835.28454596, 1999.12927118],
                 [1046.19702105, 743.46590955, 592.95869175, 524.10774715,
                  499.38476497, 514.20733799, 593.71533901, 795.06843747,
                  1088.80015664, 1319.82150622, 1615.09671452, 2615.47429544,
                  4912.85032342, 3966.45790177, 2197.99292399, 1708.58566162,
                  1698.25493903, 2016.31886622, 2229.6629306, 2262.28522815,
                  2153.13864394, 2075.5511458, 1831.67612386, 1454.3954195],
                 [1076.95739708, 767.7100562, 624.48005518, 558.85659564,
                  539.21520344, 573.39004982, 735.66330358, 1175.04610281,
                  1454.24856071, 1441.21697048, 1483.09184081, 1487.34782311,
                  1584.88762726, 1790.97665219, 2100.87661457, 3135.88287333,
                  4713.39260349, 4781.55077123, 3304.97372231, 2832.91886407,
                  2614.59946191, 2434.11538242, 2092.13811462, 1608.12465087],
                 [899.11416455, 684.02896997, 603.9178668, 582.95995188,
                  597.02198954, 686.6068089, 1234.00469333, 3534.1331761,
                  2925.28366032, 1326.86539447, 1014.67541007, 952.45652718,
                  1004.1651598, 1033.69487427, 984.78516718, 1098.29171283,
                  1332.94560983, 1725.28576874, 1808.8681104, 1860.87802118,
                  1773.77294548, 1722.19795352, 1549.18947218, 1225.38493609],
                 [1433.63696948, 960.17417448, 757.08174241, 685.69120275,
                  667.90362613, 713.51926642, 934.18354462, 1577.26653973,
                  1825.25386247, 1631.31427569, 1592.29403217, 1551.69000017,
                  1578.45990546, 1631.1542483, 1610.63678576, 1772.0194418,
                  2343.52502881, 3659.63499858, 4914.64129545, 5772.12142774,
                  5439.95091113, 4704.81967898, 3740.83264018, 2635.09949725]]

    centroids = [[np.round(float(i), 0) for i in nested] for nested in centroids]

    return centroids


def fit_kmeans_to_centroids(centroids):
    kmeans = KMeans(n_clusters=21, init=centroids, n_init=1)
    kmeans.fit(centroids)
    return kmeans


# plot centroids curves
def plot_centroids(centroids):
    df_centroids = pd.DataFrame(centroids, columns=[*range(0, 24, 1)])
    df_centroids = df_centroids.round(0)
    df_centroids = df_centroids.astype(int)
    M = 6
    N = 4
    fig, axs = plt.subplots(M, N, figsize=(15, 15))
    fig.tight_layout(pad=6.0)
    fig.suptitle('Centroids', fontsize=20)
    centroids_id = 0
    for i in range(M):
        for j in range(N):
            if centroids_id < len(centroids):
                axs[i, j].plot(df_centroids.iloc[centroids_id])
                axs[i, j].set_title("centroid " + str(centroids_id))
                centroids_id += 1


def set_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    print(device_lib.list_local_devices())
    devices = tf.config.list_physical_devices()
    print("\nDevices: ", devices)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        print("GPU details: ", details)
    tf.config.list_physical_devices()




