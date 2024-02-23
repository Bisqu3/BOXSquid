import tensorflow as tf
#player class holds individuals information
class Player:
    def __init__(self,uniqueID):
        self.id = uniqueID
        self.model = None
    
    #assign id# and compile model
    def initialize(self,learningRate,clipValue):
        self.model.compile(optimizer='adam', loss='mae', metrics=['mse'])

    #get model status
    def getModelState(self):
        self.model.summary()

