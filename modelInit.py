from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

def initializeLSTM(inputShape, neuronCount, layers=0, activation='linear', dropout_rate=0.34):
    model = Sequential()
    count = 0
    for neurons in neuronCount:
        #add LSTM layer with dropout for all but the last layer
        if count == len(neuronCount) - 1:
            model.add(LSTM(neurons, return_sequences=True, input_shape=inputShape, activation='relu'))
        else:
            model.add(LSTM(neurons, return_sequences=True, input_shape=inputShape, activation='relu'))
            model.add(Dropout(dropout_rate))  #dropout layer to reduce overfitting
        count += 1
    model.add(Dense(1, activation=activation))
    return model

def initializeFNN(inputShape):
    model = Sequential([ 
    Dense(64, activation='relu', input_shape=inputShape), 
    Dense(1, activation='linear') 
    ]) 
    return model

#confirm structure
#model = initializeLSTM((10,1))
#model.summary()
