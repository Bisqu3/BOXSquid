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
BATCH_SIZE = 128

#simulation constants
POPULATION = 1
EPOCHS = 20
playerList = {}

#model constants
LEARNING_RATE = 0.00005
CLIP_VALUE = 1.3
#add or remove for more layers with set neurons on layer. eg: [8,4] 2 layers. 1st layer 8 neurons, 2nd 4.
LAYER_NEURONS = [64,128,64]
INPUT_SHAPE = (8,8)


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
        #playerList[player].model = mi.initializeLSTM(INPUT_SHAPE, LAYER_NEURONS)
        playerList[player].model = mi.initializeFNN(INPUT_SHAPE)
        playerList[player].model.summary()
        #input("check model summary. press enter")
        playerList[player].initialize(LEARNING_RATE,CLIP_VALUE)
        results = mt.generatorModelTrain(generator_train, generator_test, playerList[player].model,EPOCHS)
        #print(results)
        #ship it to analysis
        truth = dPP.denormalize_data(y_test,target_scaler)
        predictions = mA.getAnalysis(generator_test,y_test,playerList[player].model)
        totalq = len(predictions)
        correct = [0,0,0,0,0]
        for guess in range(totalq):
            p = int(dPP.denormalize_data(predictions[guess],target_scaler)[0][0])
            a = int(truth[guess][0])
            print(f'prediction #{guess}')
            print(f'prediction: {p}')
            print(f'answer: {a}')
            print('\n\n')
            for i in range(5):
                if p >= a-i and p <= a+i:
                    correct[i] += 1
        print(f'total correct: {correct[0]}/{totalq}')
        print(f'total correct +-1: {correct[1]}/{totalq}')
        print(f'total correct +-2: {correct[2]}/{totalq}')
        print(f'total correct +-3: {correct[3]}/{totalq}')
        print(f'total correct +-4: {correct[4]}/{totalq}')

        print(f'correct percentage: {(correct[0]/totalq)*100:.2f}')
        print(f'correct percentage +-1: {(correct[1]/totalq)*100:.2f}')
        print(f'correct percentage +-2: {(correct[2]/totalq)*100:.2f}')
        print(f'correct percentage +-3: {(correct[3]/totalq)*100:.2f}')
        print(f'correct percentage +-4: {(correct[4]/totalq)*100:.2f}')
        sr = playerList[player].model.predict([[1,0.56,0.43,0.155,0.8675,0.4,0.172,0.229]])
        print(dPP.denormalize_data(sr,target_scaler))


main()