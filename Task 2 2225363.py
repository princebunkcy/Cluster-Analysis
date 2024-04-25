#!/usr/bin/env python
# coding: utf-8

# ## Task 2: Clustering
# 

# **importing necessary libraries** 

# In[49]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


# In[50]:


df = pd.read_csv('C:/Users/Hp 2022/Documents/dataset/country_data.csv')


# In[51]:


df


# In[52]:


print(df.head()) 
print(df.info()) 
print(df.corr(),'\n') 


# **remove missing rows/values**

# In[78]:


df = df.dropna(axis = 0)


# In[54]:


df


# checking for correlattion using heatmap

# In[55]:


correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))  
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")


# using two features from the table

# In[110]:


X = df.iloc[:, [2,4]].values


# In[111]:


print(X[:5])


# ploting the initial features 

# In[112]:


feature1 = 'exports'  # Replace with your feature column name
feature2 = 'imports'  # Replace with another feature column name
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

# Scatter plot of Feature1 vs Feature2
plt.scatter(df[feature1], df[feature2], s=50, c='blue', alpha=0.7, label=f'{feature1} vs {feature2}')

# Customize the plot
plt.title(f'Scatter Plot: {feature1} vs {feature2}')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
plt.tight_layout()
plt.savefig('4scatter_plot.jpg', dpi=300)  # Adjust the dpi value as needed


# In[113]:


# contruct the modelk-means 
model = KMeans(n_clusters = 2, random_state=5)
model.fit(X)
cluster_centers = model.cluster_centers_


# In[114]:


# print the centre positions of the clusters
centers = model.cluster_centers_
print('Centroids:', centers, '\n')


#Visualise the result
fig, ax = plt.subplots()

# store the normalisation of the color encodings 
# based on the number of clusters
nm = Normalize(vmin = 0, vmax = len(centers)-1)

# plot the clustered data
scatter1 = ax.scatter(X[:, 0], X[:, 1],
	c = model.predict(X), s = 50, cmap = 'plasma', norm = nm)

# plot the centroids using a for loop
for i in range(centers.shape[0]):
	ax.text(centers[i, 0], centers[i, 1], str(i), c = 'black',
		bbox=dict(boxstyle="round", facecolor='white', edgecolor='black'))


ax.set_xlabel(df.columns[2])
ax.set_ylabel(df.columns[4])


# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter1.legend_elements(),
                    loc="upper right", title="Clusters")
ax.add_artist(legend1)


fig.savefig('cluster_plot.png')


# **Predicting the modle to know the cluster and data points** 

# In[115]:


# tell algorithm the number of clusters it should look for
kmeans = KMeans(n_clusters = 2, random_state=5)
# run the Kmeans algorithm for the data X
kmeans.fit(X)


# In[116]:


test_point = np.array([[29.89007438 ,37.20219752]])
print('Prediction1:', kmeans.predict(test_point), '\n')


# In[96]:


predicted_cluster = kmeans.predict(test_point)


# using 3 features 

# In[123]:


X2 = df.iloc[:, [2,4,6]].values


# In[125]:


# contruct the model  k-means
model = KMeans(n_clusters = 6, random_state=5)
model.fit(X2)


cluster_centers = model.cluster_centers_

# print the centre positions of the clusters
centers = model.cluster_centers_
print('Centroids:', centers, '\n')



#Visualise the result in a 3D plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection = '3d')

# store the normalisation of the color encodings 
# based on the number of clusters
nm = Normalize(vmin = 0, vmax = len(centers)-1)

# plot the clustered data
scatter1 = ax.scatter(X2[:, 0], X2[:, 1], X2[:,2],
	c = model.predict(X2), s = 50, cmap = 'plasma', norm = nm)

# plot the centroids using a for loop
for i in range(centers.shape[0]):
	ax.text(centers[i, 0], centers[i, 1], centers[i, 2],
	 str(i), c = 'black',
		bbox=dict(boxstyle="round", facecolor='white',
		 edgecolor='black'))


ax.azim = -60
ax.dist = 10
ax.elev = 10

# make sure you choose the correct column names here!!!
ax.set_xlabel(df.columns[2])
ax.set_ylabel(df.columns[4])
ax.set_zlabel(df.columns[6])


# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter1.legend_elements(),
                    loc="center left", title="Clusters")
ax.add_artist(legend1)


fig.tight_layout(pad=-2.0)


fig.savefig('cluster_3Dplot.png')


# **predicting the model to know the data points and cluster of the features**

# In[130]:


# tell algorithm the number of clusters it should look for
kmeans = KMeans(n_clusters =6 , random_state=5)
# run the Kmeans algorithm for the data X
kmeans.fit(X2)
test_point = np.array([[26.9 ,17.5,74.95]])
print('Prediction:', kmeans.predict(test_point), '\n')


# testing with random_state = 500

# In[131]:


# contruct the model (either k-means or mean shift)
model = KMeans(n_clusters = 6, random_state=500)
model.fit(X2)


cluster_centers = model.cluster_centers_

# print the centre positions of the clusters
centers = model.cluster_centers_
print('Centroids:', centers, '\n')



#Visualise the result in a 3D plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection = '3d')

# store the normalisation of the color encodings 
# based on the number of clusters
nm = Normalize(vmin = 0, vmax = len(centers)-1)

# plot the clustered data
scatter1 = ax.scatter(X2[:, 0], X2[:, 1], X2[:,2],
	c = model.predict(X2), s = 50, cmap = 'plasma', norm = nm)

# plot the centroids using a for loop
for i in range(centers.shape[0]):
	ax.text(centers[i, 0], centers[i, 1], centers[i, 2],
	 str(i), c = 'black',
		bbox=dict(boxstyle="round", facecolor='white',
		 edgecolor='black'))


ax.azim = -60
ax.dist = 10
ax.elev = 10

# make sure you choose the correct column names here!!!
ax.set_xlabel(df.columns[2])
ax.set_ylabel(df.columns[4])
ax.set_zlabel(df.columns[6])


# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter1.legend_elements(),
                    loc="center left", title="Clusters")
ax.add_artist(legend1)


fig.tight_layout(pad=-2.0)


fig.savefig('11cluster_3Dplot.png')


# **carrying out prediction** 

# In[132]:


# tell algorithm the number of clusters it should look for
kmeans = KMeans(n_clusters =6 , random_state=500)
# run the Kmeans algorithm for the data X
kmeans.fit(X2)
test_point = np.array([[25.3,17.4,104]])
print('Prediction:', kmeans.predict(test_point), '\n')

