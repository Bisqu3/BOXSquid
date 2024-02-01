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
        # Compile the model if it has not been compiled
        if not hasattr(self.DataModel, 'compiled') or not self.DataModel.compiled:
            self.DataModel.compile(
                loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
            )
            self.DataModel.compiled = True

        # Train the model
        history = self.DataModel.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        return history
    
    def crossover(self, other_player):
        # Create a new model with blended weights
        new_model = self.blend_weights(self.DataModel, other_player.DataModel)
        # Create a new Player instance
        new_player = Player()
        # Assign the new model to the new player
        new_player.DataModel = new_model
        return new_player

    @staticmethod
    def blend_weights(model1, model2):
        # Create dummy input with a specific shape
        dummy_input_shape = (10, 1)  # Adjust the shape based on your input requirements
        dummy_input = tf.keras.Input(shape=dummy_input_shape)

        # Ensure models have been called with inputs to define their input shapes
        model1(dummy_input)
        model2(dummy_input)

        # Get the weights from each model
        weights1 = model1.get_weights()
        weights2 = model2.get_weights()

        # Blend the weights (for simplicity, just averaging the weights)
        blended_weights = [(w1 + w2) / 2 for w1, w2 in zip(weights1, weights2)]

        # Create a new model with the blended weights
        new_model = tf.keras.models.clone_model(model1)
        new_model.set_weights(blended_weights)

        return new_model