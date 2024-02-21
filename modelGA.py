#player class holds individuals information
class Player:
    def __init__(self,uniqueID):
        self.id = uniqueID
        self.model = None
    
    #assign id# and compile model
    def initialize(self):
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer= tf.keras.optimizers.Adam(learning_rate=learningRate, clipvalue=clipValue))

    #get model status
    def getModelState(self):
        self.model.summary()

