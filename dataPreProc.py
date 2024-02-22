import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split

def denormalize_data(data,scaler):
    return scaler.inverse_transform(data)

def read_and_preprocess_excel(file_path):
    # Read Excel file
    df = pd.read_excel(file_path, engine='openpyxl')

    # Identify input variables (all columns except the last one)
    input_variables = df.columns[:-1]
    input_data = df[input_variables]

    # Identify target variable (last column)
    target_variable = df.columns[-1]
    target_data = df[target_variable]

    # Normalize input and target data
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    target_data_scaled = scaler.fit_transform(target_data.values.reshape(-1, 1))

    # Display processed DataFrame
    print("Processed DataFrame - Input Variables:")
    print(input_data.head())
    print("\nProcessed DataFrame - Target Variable:")
    print(target_data.head())

#might need to write in another print function to display normalized data

    # Create sequences using TimeseriesGenerator #
    #NOT REALLY SURE WHAT"S GOING ON HERE, THERE's FOR SURE A BETTER WAY TO DO THIS SECTION
    n_steps = int(input("Enter the number of time steps: "))
    batch_size = int(input("Enter the batch size: "))

    #Store variable labels
    input_variable_labels = list(input_variables)
    target_variable_label = target_variable

    generator = TimeseriesGenerator(input_data_scaled, target_data_scaled, length=n_steps, batch_size=batch_size)
    dataCount = 1
    try:
        for i in generator:
            print(f"-- Series {dataCount} --\n\n")
            for j in range(batch_size):
                print(f"Batch {j+1}")
                print("Training Data:")
                print(i[0][j])
                print("Testing Data:")
                print(i[1][j])
                print("\n\n\n")
            dataCount += 1
            break
    except: print("ohno!")

    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(input_data_scaled, target_data_scaled, test_size=0.2, random_state=42)
    return generator,scaler