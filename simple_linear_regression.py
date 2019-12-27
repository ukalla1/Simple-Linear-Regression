import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('./Salary_Data.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:, 1:].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_predict_train = regressor.predict(x_train)
y_predict = regressor.predict(x_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, y_predict_train, color = 'blue')
plt.title("Salary vs Experience (training set)")
plt.xlabel("Experience in years")
plt.ylabel("Salary in USD")
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, y_predict_train, color = 'blue')
plt.title("Salary vs Experience (test set)")
plt.xlabel("Experience in years")
plt.ylabel("Salary in USD")
plt.show()