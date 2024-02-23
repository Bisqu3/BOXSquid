import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split

def denormalize_data(data,scaler):
    return scaler.inverse_transform(data)

def read_and_preprocess_excel(file_path, n_steps, batch_size):
    # Read excel file
    df = pd.read_excel(file_path, engine='openpyxl')

    input_variables = df.columns[:-1]
    input_data = df[input_variables]
    target_variable = df.columns[-1]
    target_data = df[target_variable]

    # Normalize input data
    input_scaler = MinMaxScaler()
    input_data_scaled = input_scaler.fit_transform(input_data)
    
    # Normalize target data
    target_scaler = MinMaxScaler()
    target_data_scaled = target_scaler.fit_transform(target_data.values.reshape(-1, 1))  # Reshape and scale target data

    # Display processed dataframe
    print("Processed DataFrame - Input Variables:")
    print(f"untouched\n {input_data.head()}")
    print(f"normalized\n {input_data_scaled[:5]}")
    print(f'denormalized\n {denormalize_data(input_data_scaled[:5],input_scaler)}')
    print("\nProcessed DataFrame - Target Variable:")
    print(f"untouched\n {target_data[:5]}")  # Displaying first 5 elements of target_data
    print(f"normalized\n {target_data_scaled[:5]}")
    print(f'denormalized\n {denormalize_data(target_data_scaled[:5],target_scaler)}')
    #input("dataset preview. hit enter to continue...")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_data_scaled, target_data_scaled, test_size=0.2, random_state=42)
    for x_train_sample, y_train_sample in zip(X_train, y_train):
        # Print the input and target data for the current sample
        print("X_train:", x_train_sample)
        print("y_train:", y_train_sample)
        print("---------DENORMALIZED-------")
        break
    print(denormalize_data(X_train[:1],input_scaler))
    print(denormalize_data(y_train[:1],target_scaler))
    #input("split dataset preview. hit enter to continue...")



    generator_train = TimeseriesGenerator(X_train, y_train, length=n_steps, batch_size=batch_size)
    generator_test = TimeseriesGenerator(X_test, y_test, length=n_steps, batch_size=batch_size)
    for i in generator_train:
        print(i)
        break
    print('\n\n')
    for i in generator_test:
        print(i)
        break
    #input("generator preview. hit enter to continue...")

    return generator_train, generator_test, input_scaler, target_scaler, y_test
