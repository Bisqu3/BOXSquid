from keras.models import Sequential
from keras.layers import Dense,LSTM

def initializeLSTM(inputShape,layers=0,neuronCount=64,half_life=False, activation='linear'):
    model = Sequential()
    if layers > 0:
        #multi layer LSTM
        model.add(LSTM(neuronCount, return_sequences=True , input_shape=inputShape))
        neurons = neuronCount
        for i in range(layers):
            if half_life:
                if neurons > 1:
                    neurons = neurons//2
                else: continue
            model.add(LSTM(neuronCount))
    else:
        #single layer LSTM
        model.add(LSTM(neuronCount,input_shape=inputShape))
    model.add(Dense(1,activation='linear'))
    return model

#confirm structure
#model = initializeLSTM((10,1))
#model.summary()
