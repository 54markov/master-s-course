#!/usr/bin/python
import commands
import os

os.system('ssh markov@jet.cpct.sibsutis.ru "rm -f info_nodes && pbsnodes -a >> info_nodes"')
os.system('scp markov@jet.cpct.sibsutis.ru:~/info_nodes /home/randy/WorkSpace/git/study/DCS/lab1/')
os.system('./make_table.py')
os.system('rm -f info_nodes')