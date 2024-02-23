from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
#TODO all graph plotting happens here

def getAnalysis(generator_test,y_test,model):
    # Generate predictions for the test set
    predictions = model.predict(generator_test)

    #flatten the predictions and actual values for evaluation
    flat_predictions = predictions.flatten()
    flat_actual_values = y_test.flatten()

    #TODO confirm these even line up correctly. could be completely different values. want to say yes because model.evaluate() spits similar values (mse)
    mse = mean_squared_error(flat_actual_values[:835], flat_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(flat_actual_values[:835], flat_predictions)
    r2 = r2_score(flat_actual_values[:835], flat_predictions)

    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("R^2 Score:", r2)
