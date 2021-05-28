import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv('hiring.csv')


X = df.iloc[:, :3]


y = df.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)


#saving the model to disk
pickle.dump(regressor,open("model.pkl","wb"))

#Prediction
model=pickle.load(open("model.pkl","rb"))
print(model.predict([[2,7,8]]))