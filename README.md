# Cluster - Kmeans
In this project, I applied machine learning algorithm to the country_data.csv dataset. The dataset contains information about a countries child mortality, exports, health
spending, etc. I used clustering to investigate the dataset, analyse the results and discuss what can be concluded by the clusters.

## Introduction 
While carrying out this task , I imported the necessary libraries including scikit- learn library for KMeans and imported the country dataset .I carried out some pre-processing steps just to make sure the data was ready to analyse.
To construct this model , I inserted 2 features in X (exports and imports) with respect to gdpp using the function X = df.iloc[:, [2,4]].values, I also invoked the KMeans algorithm to fit the data which Is used for unsupervised machine learning with an assumed cluster of 2 and a random state of 5 .I found the coordinate of the cluster centres (cluster_centers = model.cluster_centers_ ) , ran a test to make prediction and finally carried out a plot to visualise the model 

### Is there a way of visualising your model? (Possibly just one or two input/feature variable(s).)
In this model , I started by constructing a simple model as seen below with an assumed cluster of two with centroids at 0 and 1 and two features’ “exports” and “imports” in respects to the “gdpp”

![Diagram Title](/charts/cluster_plot.png)

### Can you make any conclusions about the clustering?
The above cluster which was carried out with KMeans clustering algorithm, shows how exports and imports varies with different labels 0 and 1 in terms of which centroids it is associates with and can facilitate the forecasting of GDP growth which is used for investment decision making.

### Include as many features as you can. Does the clustering change?
Yes , the clustering changed as I added additional feature “inflation” to my initial features with an assumed cluster = 6 and random_state = 500 which gave a better review of the cluster in respect to the gdpp

![Diagram Title](/charts/cluster_3Dplot.png)

### What advice would you give, in the context of the data, based on the clustering?
Well,to countries seeking for ways to improve on their GDP, they should be able to increase the importation and exportation rate with a decreased inflation rate .looking at above graph, we can observe that at cluster 1 when inflation was low with centroids 2.468 the exports and imports features were high and had 176 and 56.7 respectively compared to when inflation at cluster 5 had high rate of 104 as centroids , whereas export and import had low rate of 25.3 and 17.4 
