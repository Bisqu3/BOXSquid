#brings together all scripts for current use in terminal
#model scripts
import modelInit as mi
import modelTrain as mt
import modelGA as mga
#data scripts
import dataPreProc as dPP

#TODO complete modules and assemble here lol
POPULATION = 10
playerList = {}


def main():
    BASE_NAME = "player"
    for i in range(POPULATION):
        uniqueName = BASE_NAME+str(i+1)
        playerList[uniqueName] = mga.Player(uniqueName)
    for player in playerList:
        #how to call on model in dic
        print(playerList[player].id)

main()