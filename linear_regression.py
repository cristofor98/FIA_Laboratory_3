import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = pd.read_csv('apartmentComplexData.txt', sep=",", header=None)
data.isna().sum()
print('Let\'s check for null values\n')
print(data.isnull().sum())

data.columns = ['column1','column2','complexAge','totalRooms','totalBedrooms','complexInhabitants','apartmentsNr','column8','medianCompexValue']
x = data[['column1','column2','complexAge','totalRooms','totalBedrooms','complexInhabitants','apartmentsNr','column8']]
y = data['medianCompexValue']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.40,random_state=42)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train)
print('Intercept :', round(lm.intercept_,2))
print('Slope :', round(lm.coef_[0],2))
coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
predictions = lm.predict(x_test)
from sklearn.metrics import r2_score
score = r2_score(y_test,predictions)
print(score)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,predictions)
print(mse)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test,predictions)
print(mae)

filename = 'model/apartment_model.pkl'
joblib.dump(lm,filename)

