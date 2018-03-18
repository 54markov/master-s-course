#!/usr/bin/python
import commands
import os

os.system('ssh markov@jet.cpct.sibsutis.ru "qsub dcs/task.job"')
os.system('./state_system.py')