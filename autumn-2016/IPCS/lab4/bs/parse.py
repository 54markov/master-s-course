#!/usr/bin/python
import commands
import os
from terminaltables import AsciiTable

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

table_data = []

table_data.append(['Type', 'Result', 'Score', 'Description'])

def measureDescription(line):
    x = float(line)
    if x < 0.00016:
        return bcolors.OKGREEN + "Random" + bcolors.ENDC
    elif x > 0.00016 and x < 0.00393:
        return bcolors.OKGREEN + "Random" + bcolors.ENDC
    elif x > 0.00393 and x < 0.1015:
        return bcolors.OKGREEN + "Random" + bcolors.ENDC
    elif x > 0.1015 and x < 0.4549:
        return bcolors.OKBLUE + "Prety much random" + bcolors.ENDC
    elif x > 0.4549 and x < 1.323:
        return bcolors.OKBLUE + "Allmost random" + bcolors.ENDC
    elif x > 1.323 and x < 2.7055:
        return bcolors.WARNING + "May be random" + bcolors.ENDC
    elif x > 2.7055 and x < 3.8415:
        return bcolors.WARNING + "Non-random" + bcolors.ENDC
    elif x > 3.8415 and x < 5.0239:
        return bcolors.FAIL + "Non-random" + bcolors.ENDC
    elif x > 5.0239 and x < 7.8794:
        return bcolors.FAIL + "Non-random" + bcolors.ENDC
    elif x > 7.8794 and x < 10.8276:
        return bcolors.FAIL + "Non-random" + bcolors.ENDC

    return bcolors.FAIL + "Non-random" + bcolors.ENDC

def measureScore(line):
    x = float(line)
    if x < 0.00016:
        return bcolors.OKGREEN + "0.99" + bcolors.ENDC
    elif x > 0.00016 and x < 0.00393:
        return bcolors.OKGREEN + "0.95" + bcolors.ENDC
    elif x > 0.00393 and x < 0.1015:
        return bcolors.OKGREEN + "0.75" + bcolors.ENDC
    elif x > 0.1015 and x < 0.4549:
        return bcolors.OKBLUE + "0.50" + bcolors.ENDC
    elif x > 0.4549 and x < 1.323:
        return bcolors.OKBLUE + "0.25" + bcolors.ENDC
    elif x > 1.323 and x < 2.7055:
        return bcolors.WARNING + "0.10" + bcolors.ENDC
    elif x > 2.7055 and x < 3.8415:
        return bcolors.WARNING + "0.05" + bcolors.ENDC
    elif x > 3.8415 and x < 5.0239:
        return bcolors.FAIL + "0.010" + bcolors.ENDC
    elif x > 5.0239 and x < 7.8794:
        return bcolors.FAIL + "0.005" + bcolors.ENDC
    elif x > 7.8794 and x < 10.8276:
        return bcolors.FAIL + "0.001" + bcolors.ENDC

    return bcolors.FAIL + "0.0001" + bcolors.ENDC

with open("file") as file:
    for line in file:
        rc = line.find("./bs")
        if rc >= 0:
            table_data.append([line, ' '])

        rc = line.find("x^2")
        if rc >= 0:
            rc = line.find("=")
            score = measureScore(line[rc+1:-1])
            description = measureDescription(line[rc+1:-1])
            table_data.append(['', line, "a = " + score, description])

table = AsciiTable(table_data)
print table.table
