#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""1)Main objective of the analysis that also specifies whether your model 
will be focused on clustering or dimensionality reduction and the benefits that 
your analysis brings to the business or stakeholders of this data.

The main objective is to create a clustering algorithms to group dataset according to the tenure of the credit card and 
then apply the clustering algorithm between two highly coorolated 
features to see how clusters are created (visual interpretation). 

This clustering techniques for this dataset could be used to define a marketing strategies by using customer segmentation. """ 


# In[ ]:


"""2. Dataset description:
Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months.
The file is at a customer level with 18 behavioral variables.
Following is the Data Dictionary for Credit Card dataset :-

CUSTID : Identification of Credit Card holder (Categorical)
BALANCE : Balance amount left in their account to make purchases (
BALANCEFREQUENCY : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
PURCHASES : Amount of purchases made from account
ONEOFFPURCHASES : Maximum purchase amount done in one-go
INSTALLMENTSPURCHASES : Amount of purchase done in installment
CASHADVANCE : Cash in advance given by the user
PURCHASESFREQUENCY : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
ONEOFFPURCHASESFREQUENCY : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
PURCHASESINSTALLMENTSFREQUENCY : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
CASHADVANCEFREQUENCY : How frequently the cash in advance being paid
CASHADVANCETRX : Number of Transactions made with "Cash in Advanced"
PURCHASESTRX : Numbe of purchase transactions made
CREDITLIMIT : Limit of Credit Card for user
PAYMENTS : Amount of Payment done by user
MINIMUM_PAYMENTS : Minimum amount of payments made by user
PRCFULLPAYMENT : Percent of full payment paid by user
TENURE : Tenure of credit card service for user """


# In[33]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import seaborn as sns, pandas as pd, numpy as np


# In[34]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
# Importing Library:



# In[35]:


data = pd.read_csv("CC GENERAL.csv")


# In[36]:


data.head(10)


# In[ ]:


#We can see distribution in feature TENURE that will be used for segmantation


# In[37]:


data['TENURE'].value_counts()


# In[ ]:


# Data cleaning for this dataset required: 
# - dropping/ ignoring an object feature CUST_ID
# - replacing all Nan values with 0 to be able to fit models 
# - scaling data to improve results 


# In[38]:


data.dtypes


# In[39]:


data.info()


# In[40]:


data.describe()


# In[41]:


data.isnull().sum().sort_values(ascending=False)


# In[42]:


data = data.fillna(0)


# In[43]:


data.isnull().sum().sort_values(ascending=False)


# In[ ]:





# In[44]:


### BEGIN SOLUTION
float_columns = [x for x in data.columns if x not in ['CUST_ID']]

# The correlation matrix
corr_mat = data[float_columns].corr()

# Strip out the diagonal values for the next step
for x in range(len(float_columns)):
    corr_mat.iloc[x,x] = 0.0
    
corr_mat


# In[45]:


# Pairwise maximal correlations
corr_mat.abs().idxmax()


# In[47]:


skew_columns = (data[float_columns]
                .skew()
                .sort_values(ascending=False))

skew_columns = skew_columns.loc[skew_columns > 0.75]
skew_columns


# In[48]:


# Perform log transform on skewed columns
for col in skew_columns.index.tolist():
    data[col] = np.log1p(data[col])


# In[16]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
data[float_columns] = sc.fit_transform(data[float_columns])

data.head(4)


# In[ ]:


I used kmeans and agglomerative clustering. The results are presented below with comparisons in dataframe form. 


# In[53]:


from sklearn.cluster import KMeans
### BEGIN SOLUTION
km = KMeans(n_clusters=2, random_state=42)
km = km.fit(data[float_columns])

data['kmeans'] = km.predict(data[float_columns])


# In[54]:


(data[['TENURE','kmeans']]
 .groupby(['kmeans','TENURE'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))
### END SOLUTION


# In[55]:


data['kmeans'].value_counts()


# In[56]:


data['TENURE'].value_counts()


# In[57]:


### BEGIN SOLUTION
# Create and fit a range of models
km_list = list()

for clust in range(1,21):
    km = KMeans(n_clusters=clust, random_state=42)
    km = km.fit(data[float_columns])
    
    km_list.append(pd.Series({'clusters': clust, 
                              'inertia': km.inertia_,
                              'model': km}))


# In[58]:


plot_data = (pd.concat(km_list, axis=1)
             .T
             [['clusters','inertia']]
             .set_index('clusters'))

ax = plot_data.plot(marker='o',ls='-')
ax.set_xticks(range(0,21,2))
ax.set_xlim(0,21)
ax.set(xlabel='Cluster', ylabel='Inertia');
### END SOLUTION


# In[59]:


from sklearn.cluster import AgglomerativeClustering
### BEGIN SOLUTION
ag = AgglomerativeClustering(n_clusters=2, linkage='ward', compute_full_tree=True)
ag = ag.fit(data[float_columns])
data['agglom'] = ag.fit_predict(data[float_columns])


# In[64]:




# First, for Agglomerative Clustering:
(data[['TENURE','agglom','kmeans']]
 .groupby(['TENURE','agglom'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))


# In[65]:




# Comparing with KMeans results:
(data[['TENURE','agglom','kmeans']]
 .groupby(['TENURE','kmeans'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))


# In[66]:


# Comparing results:
(data[['TENURE','agglom','kmeans']]
 .groupby(['TENURE','agglom','kmeans'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))


# In[67]:


data['agglom'].value_counts()


# In[68]:


# First, we import the cluster hierarchy module from SciPy (described above) to obtain the linkage and dendrogram functions.
from scipy.cluster import hierarchy

Z = hierarchy.linkage(ag.children_, method='ward')

fig, ax = plt.subplots(figsize=(15,5))


den = hierarchy.dendrogram(Z, orientation='top', 
                           p=30, truncate_mode='lastp',
                           show_leaf_counts=True, ax=ax)
### END SOLUTION


# In[ ]:





# In[ ]:


# Here I prepared another experiment where I used Kmeans algorithm to plot the two highly correlated algorithms.
# The main objective here is to visulise how data segmentation is performed and how clusters looks like. 


# In[87]:


#drop object value
data = pd.read_csv("CC GENERAL.csv")
data = data.drop("CUST_ID", axis=1)


# In[88]:


data.info()


# In[89]:


data = data.fillna(0)


# In[90]:


data.hist(bins=13, figsize=(20, 15), layout=(5, 4))


# In[91]:


corrmat = data.corr()
  
f, ax = plt.subplots(figsize =(20, 9))
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1, annot = True) # Annot helps in putting corelation numbers in the plotted boxes


# In[92]:


X = data.iloc[:,[12,13]]
X.head()


# In[ ]:


# I used elbow method to find optimal number of clusters suitable for this example. 


# In[93]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[94]:


kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
X = X.values


# In[ ]:


# Analysis:

# From the sequence of left to right, the first centrod implies that customers where both Payments and Credit Limit are low 
# therefore these customers dont have a large creditiability. 
# Second centroid shows that there is an improvement but our main objectives still remain low. 
# Third centroid is our target customers with high creditiability that pay on time therefore their payments are still relatively 
# low to their credit limit. 
# Fourth centroid are customers with high credit limits and high payments. This means that bank probably earns a lot on intrest 
# but might not be as reliable as customers in third centroid. 


# In[95]:


plt.figure(figsize=(18, 8), dpi=80)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Credit Limit')
plt.ylabel('Payments')
plt.legend()
plt.show()

