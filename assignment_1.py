# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 11:21:39 2021

@author: ABHISHEK
"""

import pandas as pd
df=pd.read_csv("house_rental.csv")

x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

nnr=KNeighborsRegressor(n_neighbors=1)
nnr.fit(x_train,y_train)

print(nnr.score(x_test,y_test))

rmse_value=[]
for k in range(1,20):
    nn_model=KNeighborsRegressor(n_neighbors=k)
    nn_model.fit(x_train,y_train)
    y_pred=nn_model.predict(x_test)
    rmse=sqrt(mean_squared_error(y_test, y_pred))
    rmse_value.append(rmse)
    print("RMSE value = ",rmse,"--k ->",k)
    
curve=pd.DataFrame(rmse_value)
curve.plot()