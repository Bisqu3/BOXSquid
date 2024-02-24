from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def getAnalysis(generator_test, y_test, model):
    
    predictions = model.predict(generator_test)
    return predictions
