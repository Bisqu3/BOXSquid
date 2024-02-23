#brings together all scripts for current use in terminal
#model scripts
import modelInit as mi
import modelTrain as mt
import modelGA as mga
import modelAnalysis as mA
#data scripts
import dataPreProc as dPP
import numpy as np
#data preproc constants
N_STEPS = 1
BATCH_SIZE = 64

#simulation constants
POPULATION = 1
EPOCHS = 1000
playerList = {}

#model constants
LEARNING_RATE = 0.00001
CLIP_VALUE = 2.0
LAYER_NEURONS = [8,4]


def main():
    #timeframe data generation
    generator_train, generator_test, input_scaler, target_scaler, y_test = dPP.read_and_preprocess_excel("abalone.xlsx",N_STEPS,BATCH_SIZE)
    #player assignment
    BASE_NAME = "player"
    for i in range(POPULATION):
        uniqueName = BASE_NAME+str(i+1)
        playerList[uniqueName] = mga.Player(uniqueName)
    #player model initialization
    for player in playerList:
        #how to call on model in dic
        print(playerList[player].id)
        print("Train generator data shape:", generator_train.data.shape)
        print("Test generator data shape:", generator_test.data.shape)
        #(8,8) refers to 8 different datatypes that are unique from eachother(8 columns, 8 different types).
        playerList[player].model = mi.initializeLSTM((8,8), LAYER_NEURONS)
        #RNN likely broken rn
        #playerList[player].model = mi.initializeRNN()
        playerList[player].initialize(LEARNING_RATE,CLIP_VALUE)
        results = mt.generatorModelTrain(generator_train, generator_test, playerList[player].model,EPOCHS)
        #print(results)
        #ship it to analysis
        mA.getAnalysis(generator_test,y_test,playerList[player].model)


main()