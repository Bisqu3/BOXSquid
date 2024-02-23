import tensorflow as tf
from keras.optimizers import Adam
#player class holds individuals information
class Player:
    def __init__(self,uniqueID):
        self.id = uniqueID
        self.model = None
    
    #assign id# and compile model
    def initialize(self,learningRate,clipValue):
        optimizer = Adam(learning_rate=learningRate, clipvalue=clipValue)
        self.model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])

    #get model status
    def getModelState(self):
        self.model.summary()

