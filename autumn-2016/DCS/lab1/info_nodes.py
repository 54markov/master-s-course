#!/usr/bin/python
import commands
import os

os.system('rm -f info_nodes')
os.system('pbsnodes -a >> info_nodes')

with open("info_nodes") as file:
    for line in file:
        rc = line.find("cn")
        if rc == 0:
            print line

        rc = line.find("state")
        if rc > 0 and rc < 10:
            print line

        rc = line.find("power_state")
        if rc > 0 and rc < 10:
            print line

        rc = line.find("jobs")
        if rc > 0 and rc < 10:
            print line