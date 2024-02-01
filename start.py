import tensorflow as tf
import pandas as pd
import numpy as np
import populate as pp
#MAIN SCRIPT OVERSEES DATA AND MODEL SAVING.
#for model and training options go to populate.py
Data = pd.read_csv('abalone.csv',names=['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings'],)

#TODO constants to be replaced with users input/save file
#Sim Options
POPULATION = 100
GENERATIONS = 20
#Training Options
EPOCHS = 25
LEARNING_RATE = 2.8
CLIP_VALUE = 2.5
OPTIMIZER = True
SEQUENCE_MAX = 24

#unique id 'player#' to retrieve model from class instance
activePopulationID = []
#models saved from each individual at end of round in a generation.
activePopulationModels = []

#Generation Analysis Lists
activePopulationLossHistory = []


#TODO
Data['Length'] = pd.to_numeric(Data['Length'], errors='coerce')
Data['Diameter'] = pd.to_numeric(Data['Diameter'], errors='coerce')
Data['Height'] = pd.to_numeric(Data['Height'], errors='coerce')
Data['Whole weight'] = pd.to_numeric(Data['Whole weight'], errors='coerce')
Data['Shucked weight'] = pd.to_numeric(Data['Shucked weight'], errors='coerce')
Data['Viscera weight'] = pd.to_numeric(Data['Viscera weight'], errors='coerce')
Data['Shell weight'] = pd.to_numeric(Data['Shell weight'], errors='coerce')
Data['Rings'] = pd.to_numeric(Data['Rings'], errors='coerce')
Data['Male'] = (Data['Sex']=='M').astype(int)
Data['Female'] = (Data['Sex']=='F').astype(int)
Data['Infant'] = (Data['Sex']=='I').astype(int)
#drop all NaN values
Data = Data.dropna()
#check dtype
print(Data.dtypes)
#copy main table to features
Feature = Data.copy()
# drop sex and rings from feature (sex is a string replaced by ints above, rings is what the ai is predicting.)
#labels is the answer sheet and feature is the provided data
Feature.pop('Sex')
Labels = Feature.pop('Rings')
#numpy array before feeding into tensorflow
Feature = np.array(Feature, dtype=float)
Labels = np.array(Labels, dtype=float)

#initialize individual players. MUST REMAIN OUTSIDE MAIN LOOP
baseName = "player"
activePopulationID = []

for i in range(0, POPULATION):
    uniqueName = baseName + str(i)
    print(uniqueName)
    
    # Create Player instance
    player = pp.Player()
    
    # Check for existing model progress
    try:
        player.DataModel = tf.keras.models.load_model('model.h5')
        print("Using saved model. Move save if you do not want this.")
    except:
        player.initializeModel(i, LEARNING_RATE, CLIP_VALUE)
        print('Model Initialized...')
    
    activePopulationID.append(player)

#begin loop to play through set # of generations
curRound = 1
#has to equal # of columns
sequence_length = 10
#main generation loop
while curRound <= GENERATIONS:
    print(f"ROUND {curRound} out of {GENERATIONS}")
    #local analysis by generation
    activePopulationLoss = []
    #iterate through each players round.
    for individual in activePopulationID:
        #temp to hold active model
        activeModel = None
        #print(individual,"Has started training...")
        Feature_reshaped = np.reshape(Feature, (Feature.shape[0], sequence_length, 1))
        #model is trained.
        print(d"{individual} is playing")
        activeModel = individual.train(Feature_reshaped, Labels, epochs=EPOCHS)
        #ensure model was returned to temp var
        if(activeModel != None):
            activePopulationModels.append(activeModel)
        else: print("Error model not found")
        #collect data
        activePopulationLoss.append(min(activeModel.history['loss']))
        #TODO

    #assess generation. outputs individual that had best round
    #TODO elites
    genWinner = None
    for index, individual in enumerate(activePopulationLoss):
        findLowest = min(activePopulationLoss)
        if(individual == findLowest):
            genWinner = activePopulationID[index]
            print(f"{genWinner} had the best round with {individual:.3f}")
            activePopulationLossHistory.append(individual)

    #make all active players models equal the winning model
            #TODO Mix weights and biases if enabled
    if OPTIMIZER:
        for player in activePopulationID:
            if player != genWinner:
                returnModel = genWinner.crossover(player)
                player.DataModel = returnModel.DataModel
    else:
        for player in activePopulationID:
            if player != genWinner:
                player.DataModel = genWinner.DataModel
                #print(f"{player} has been replaced!")
    #end conditions
    curRound += 1


activePopulationID[0].DataModel.save('model.h5')
print("Done.\n\n")
print(activePopulationLossHistory)
results = activePopulationID[0].DataModel.predict(Feature)
for i in range(0, len(results)):
    print(Feature[i], results[i])
