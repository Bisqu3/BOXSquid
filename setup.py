import os

file = open('requirements.txt', 'r')
for line in file:
    os.system(line)
