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

table_data.append(['Node', 'State', 'Type'])

with open("info_nodes") as file:
    for line in file:
        rc = line.find("cn")
        if rc == 0:
            table_data.append([line, ' '])
            #print line

        rc = line.find("state")
        if rc > 0 and rc < 10:
            string = line[line.find("=")+2:]
            if string.find('free') >= 0:
                string = bcolors.OKGREEN + 'Free' + bcolors.ENDC
            else:
                string = bcolors.FAIL + 'Down' + bcolors.ENDC

            table_data.append([' ', 'state', string])
            #print line

        rc = line.find("power_state")
        if rc > 0 and rc < 10:
            string = line[line.find("=")+2:]
            if string.find('Run') >= 0:
                string = bcolors.OKBLUE + 'Running' + bcolors.ENDC
            else:
                string = bcolors.OKBLUE + 'Not running' + bcolors.ENDC

            table_data.append([' ', 'power_state', string])
            #print line

        rc = line.find("jobs")
        if rc > 0 and rc < 10:
            string = line[line.find("=")+2:-1]
            string = bcolors.BOLD + string + bcolors.ENDC
            table_data.append([' ', 'job' , string])
            #print line

table = AsciiTable(table_data)
print table.table