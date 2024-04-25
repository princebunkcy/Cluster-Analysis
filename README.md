# Cluster - Kmeans
In this project I applied machine learning algorithm to the country_data.csv dataset.This data set contains information about a countries child mortality, exports, health
spending, etc.Use clustering to investigate this data set, analyse the results and discuss what can be concluded by the clusters.

## Introduction 
While carrying out this task , I imported the necessary libraries including scikit- learn library for KMeans and imported the country dataset .I carried out some pre-processing steps just to make sure the data was ready to analyse.
To construct this model , I inserted 2 features in X (exports and imports) with respect to gdpp using the function X = df.iloc[:, [2,4]].values, I also invoked the KMeans algorithm to fit the data which Is used for unsupervised machine learning with an assumed cluster of 2 and a random state of 5 .I found the coordinate of the cluster centres (cluster_centers = model.cluster_centers_ ) , ran a test to make prediction and finally carried out a plot to visualise the model 

### Is there a way of visualising your model? (Possibly just one or two input/feature variable(s).)
In this model , I started by constructing a simple model as seen below with an assumed cluster of two with centroids at 0 and 1 and two features’ “exports” and “imports” in respects to the “gdpp”

![Diagram Title](/charts/prices.png)
