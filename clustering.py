import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Data loading in progress
data = pd.read_csv('world_bank_co2_emission.csv', skiprows=4)
data.head(2)

# check missing _vlaues
data.isnull().sum()

# Impute missing values
data = data.fillna(data.mean(numeric_only=True))
data.drop(data.iloc[:, 4:35], inplace=True, axis=1)

# Omit vacant columns
data.drop(['2020', '2021'], inplace=True , axis=1)

# Select the variables of interest
variables = ['CO2 emissions (metric tons per capita)', 'CO2 emissions (kg per PPP $ of GDP)','CO2 emissions from solid fuel consumption (kt)']

# Subset the data
data = data[(data['Indicator Name'].isin(variables))]
data.columns

data.isnull().sum()

# select relevant columns
cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code',
        '1991', '1995', '2000', '2005', '2010', '2015', '2019']

# create a new dataframe with selected columns
df = data[cols].copy()

# set country name as index
df.set_index('Country Name', inplace=True)

# select columns for normalization
norm_cols = ['1991', '1995', '2000', '2005', '2010', '2015', '2019']

# normalize the data
scaler = StandardScaler()
df[norm_cols] = scaler.fit_transform(df[norm_cols])

cols = ['Country Code', 'Indicator Name', 'Indicator Code',
        '1991', '1995', '2000', '2005', '2010', '2015', '2019']

# apply KMeans clustering
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(df[norm_cols])

# display original values for each cluster
for i in range(kmeans.n_clusters):
    print(f'Cluster {i}:')
#     print(df.columns)
    print(df[df['Cluster'] == i][cols])

from sklearn.metrics import silhouette_score

score = silhouette_score(df[norm_cols], df['Cluster'])
print(f"Silhouette score: {score}")

# plot cluster membership and cluster centers
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(df[norm_cols[0]], df[norm_cols[-1]], c=df['Cluster'])
centers = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centers[:, 0], centers[:, -1], marker='x', s=200, linewidths=3, color='r')
ax.set_xlabel('1960')
ax.set_ylabel('2020')
ax.set_title('KMeans Clustering of Climate Data')
plt.show()

data.columns

# Cluster by country

# load the climate data
climate_data = pd.read_csv('world_bank_co2_emission.csv', skiprows=4)
climate_data = climate_data.fillna(climate_data.mean(numeric_only=True))

# select the columns for analysis
cols = ['1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999',
       '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008',
       '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
       '2018', '2019',]
data = climate_data[cols]
data

# standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

scaled_data

# perform KMeans clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42).fit(scaled_data)

# add cluster labels to the original data
climate_data['Cluster'] = kmeans.labels_

# print the number of countries in each cluster
print(climate_data.groupby('Cluster')['Country Name'].count())

# select one country from each cluster
sample_countries = climate_data.groupby('Cluster').apply(lambda x: x.sample(1))

# compare countries from one cluster
cluster_0 = climate_data[climate_data['Cluster'] == 0]
print(cluster_0[cols].mean())



# compare countries from different clusters
cluster_1 = climate_data[climate_data['Cluster'] == 1]
print(cluster_1[cols].mean())



# investigate trends
trend_cluster_0 = cluster_0[cols].mean()
trend_cluster_1 = cluster_1[cols].mean()
print('Trend similarity between cluster 0 and cluster 1:', np.corrcoef(trend_cluster_0, trend_cluster_1)[0,1])
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# create a scatter plot of the first two principal components
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
colors = ['blue', 'green', 'red', 'purple', 'orange']
for i in range(5):
    plt.scatter(pca_data[kmeans.labels_==i,0], pca_data[kmeans.labels_==i,1], color=colors[i])
plt.title('Scatter Plot of Climate Data by Cluster')
plt.xlabel('Principal Components')
# plt.ylabel('Principal Component Two')

plt.show()