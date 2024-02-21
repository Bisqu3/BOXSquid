#populate script contains class for individual.
#each individual will go 100 rounds. the set amount of winners with the lowest loss will share weights to create new indivudals until population is met again.

import tensorflow as tf
from tensorflow.keras import layers
normalize = layers.Normalization()


class Player:
    def __init__(self):
        self.id = None
        #model creation. 8 datatypes to 32 hidden layers TODO confirm this is correct structure?
        self.DataModel = tf.keras.Sequential([
            normalize,
            layers.Dense(1)
            ])

    def initializeModel(self, uniqueID,learningRate,clipValue):
            self.id = "player"+str(uniqueID)
            self.DataModel.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer= tf.keras.optimizers.Adam(learning_rate=learningRate, clipvalue=clipValue))
    
    #TODO does not support other datatables
    def train(self, X_train, y_train, epochs=10, batch_size=4000):
        history = self.DataModel.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        return history