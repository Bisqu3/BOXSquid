from keras.models import Sequential
from keras.layers import Dense,LSTM,BatchNormalization,Dropout

def initializeLSTM(inputShape,neuronCount, layers=0, activation='linear'):
    model = Sequential()
    count = 0
    for neurons in neuronCount:
        if count == 0:
            model.add(LSTM(neurons, return_sequences=True,input_shape=inputShape, activation='relu'))
        else:
            model.add(LSTM(neurons, return_sequences=True, activation='relu'))
        count += 1
    model.add(Dense(1,activation=activation))
    return model

def initializeRNN():
    model = Sequential([ 
    Dense(256, activation='relu', input_shape=(8,)), 
    Dense(256, activation='relu'), 
    Dropout(0.3),  
    Dense(1, activation='relu') 
    ]) 
    return model

#confirm structure
#model = initializeLSTM((10,1))
#model.summary()
