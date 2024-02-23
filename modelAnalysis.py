from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def getAnalysis(generator_test, y_test, model):
    
    predictions = model.predict(generator_test)
    flat_predictions = predictions.flatten()
    flat_actual_values = y_test.flatten()

    #evaluate
    mse = mean_squared_error(flat_actual_values[:835], flat_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(flat_actual_values[:835], flat_predictions)
    r2 = r2_score(flat_actual_values[:835], flat_predictions)
    #print analysis results
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("R^2 Score:", r2)
    print("run tensorboard --logdir=./logs for more info")
