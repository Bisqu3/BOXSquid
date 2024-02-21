import tensorflow as tf
import pandas as pd
import numpy as np
import populate as pp
#MAIN SCRIPT OVERSEES DATA AND MODEL SAVING.
#for model and training options go to populate.py
Data = pd.read_csv('abalone.csv',names=['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings'],)

#Sim Options
POPULATION = 2
GENERATIONS = 5
#Training Options
EPOCHS = 10
LEARNING_RATE = 0.16
CLIP_VALUE = 1.0

#unique id 'player#' to retrieve model from class instance
activePopulationID = []
#models saved from each individual at end of round in a generation.
activePopulationModels = []

#Generation Analysis Lists
activePopulationLossHistory = []
#TODO

#Fix DataTypes
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
sequence_length = 10
Feature_reshaped = np.reshape(Feature, (Feature.shape[0], sequence_length, 1))


#initialize individual players. MUST REMAIN OUTSIDE MAIN LOOP
baseName = "player"
for i in range(0,POPULATION):
    uniqueName = baseName+str(i)
    print(uniqueName)
    makeBot = f"{uniqueName} = pp.Player()"
    exec(makeBot)
    #TODO check for existing model progress
    try:
        initializeBot = f"{uniqueName}.DataModel = tf.keras.models.load_model('model.h5')"
        exec(initializeBot)
        print("using saved model. move save if you do not want this.")
    except:
        initializeBot = f"{uniqueName}.initializeModel(i,{LEARNING_RATE},{CLIP_VALUE})"
        exec(initializeBot)
        print('Model Initialized...')
    exec(initializeBot)
    activePopulationID.append(uniqueName)

#begin loop to play through set # of generations
curRound = 1
while curRound <= GENERATIONS:
    print(f"ROUND {curRound} out of {GENERATIONS}")
    #local analysis by generation
    activePopulationLoss = []
    #iterate through each players round.
    for individual in activePopulationID:
        #temp to hold active model
        activeModel = None
        #print(individual,"Has started training...")
        #takes unique id and makes a string of the code to execute.
        getModel = f"activeModel = {individual}.train(Feature_reshaped,Labels,epochs={EPOCHS})"
        #execute string. model is trained.
        exec(getModel)
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
            genWinner = f"player{index}"
            print(f"{genWinner} had the best round with {individual:.3f}")
            activePopulationLossHistory.append(individual)

    #make all active players models equal the winning model
            #TODO Mix weights and biases if enabled
    for player in activePopulationID:
        if player != genWinner:
            saveGenWinner = f"{player}.DataModel = {genWinner}.DataModel" 
            #print(f"{player} has been replaced!")
    #end conditions
    curRound += 1


exec(f"player0.DataModel.save('model.h5')")
print("Done.\n\n")
print(activePopulationLossHistory)
exec(f"results = player0.DataModel.predict(Feature_reshaped)")
for i in range(0,len(results)):
    print(Feature[i],results[i])
