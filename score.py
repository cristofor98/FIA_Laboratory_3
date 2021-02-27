import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from azureml.core.model import Model


def init():
    global model
    model_path = Model.get_model_path('apartment_model')
    model = joblib.load(model_path)

def run(raw_data):
	data = np.array(json.loads(raw_data)['data'])
	y_hat = model.predict(data)
    return json.dumps(y_hat.tolist())


