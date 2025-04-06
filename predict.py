import pickle
import numpy as np

with open("sales_model.pkl", "rb") as file:
    model = pickle.load(file)

sample_input = np.array([[150.0, 20.0, 15.0]])
prediction = model.predict(sample_input)
print(f"Predicted Sales: {prediction[0]:.2f}")
