import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Adversing.csv")

X = df.drop("Sales", axis=1)
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

with open("sales_model.pkl", "wb") as file:
    pickle.dump(model, file)

y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
