# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 11:29:03 2021

@author: ABHISHEK
"""

import pandas as pd
df=pd.read_csv("house_rental.csv")

x=df.iloc[:,1:].values

print(df.isnull().sum())
print(df.isna().sum())

from sklearn.cluster import KMeans
k_means=KMeans(n_clusters=4,init='k-means++',random_state=0)
k_means.fit(x)

print(k_means.labels_) 

wcss=[]

for k in range(1,11):
    km=KMeans(n_clusters=k,init='k-means++')
    km.fit(x)
    wcss_i=km.inertia_
    wcss.append(wcss_i)
    print(k,wcss_i)
    
import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()