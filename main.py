#brings together all scripts for current use in terminal
#model scripts
import modelInit as mi
import modelTrain as mt
import modelGA as mga
#data scripts
import dataPreProc as dPP

#TODO complete modules and assemble here lol
POPULATION = 1
playerList = {}

LEARNING_RATE = 0.02
CLIP_VALUE = 0.1


def main():
    #timeframe data generation
    generator,scaler = dPP.read_and_preprocess_excel("abalone.xlsx")
    #player assignment
    BASE_NAME = "player"
    for i in range(POPULATION):
        uniqueName = BASE_NAME+str(i+1)
        playerList[uniqueName] = mga.Player(uniqueName)
    #player model initialization
    for player in playerList:
        #how to call on model in dic
        print(playerList[player].id)
        print("Generator data shape:", generator.data.shape)
        playerList[player].model = mi.initializeLSTM((generator.length,generator.data.shape[1]), neuronCount = 8)
        playerList[player].initialize(LEARNING_RATE,CLIP_VALUE)
        results = mt.generatorModelTrain(generator,playerList[player].model)
        print(dPP.denormalize_data(results,scaler))


main()