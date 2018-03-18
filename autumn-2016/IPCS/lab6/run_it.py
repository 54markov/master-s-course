#!/usr/bin/python
import commands
import string
import os
from terminaltables import AsciiTable

ORIGINAL_END = " transform" 
ORIGINAL_TEXT= "In computer science, an FM-index is a compressed full-text substring index based on the Burrows-Wheeler"
FAKE_END = " fake end"

def generateByLeters():
    s = string.ascii_lowercase
    for i in s[s.index('a'):s.index('z')+1]:
        cmd = "echo " + ORIGINAL_TEXT + " " + i + " > text/" + i + ".txt"
        os.system(cmd)


def generateReal():
    cmd = "echo " + ORIGINAL_TEXT + " " + ORIGINAL_END + " > text/original.txt"
    os.system(cmd)


def generateFake():
    cmd = "echo " + ORIGINAL_TEXT + " " + FAKE_END + " > text/fake.txt"
    os.system(cmd)


def runTest():
    s = string.ascii_lowercase
    for i in s[s.index('a'):s.index('z')+1]:
        cmd = "./main text/" + i + ".txt"
        os.system(cmd)

    cmd = "./main text/fake.txt"
    os.system(cmd)
    cmd = "./main text/original.txt"
    os.system(cmd)

def gatherTar():
    s = string.ascii_lowercase
    for i in s[s.index('a'):s.index('z')+1]:
        cmd = "tar -zcvf result/" + i + ".txt.tar.gz text/" + i + ".txt"
        os.system(cmd)

    cmd = "tar -zcvf result/original.txt.tar.gz text/original.txt"
    os.system(cmd)
    cmd = "tar -zcvf result/fake.txt.tar.gz text/fake.txt"
    os.system(cmd)


def gatherResult():
    os.system("ls -l result/ > statistic")

    table_data = []

    table_data.append(['FILE', 'SIZE_MY', 'SIZE_TARGZ'])

    with open("statistic") as file:
        for line in file:
            rc = line.find("compress_file")
            if rc > 0:
                size = line
                rc1 = size.find(" ")
                size = size[rc1+1:]
                rc1 = size.find(" ")
                size = size[rc1+1:]
                rc1 = size.find(" ")
                size = size[rc1+1:]
                rc1 = size.find(" ")
                size = size[rc1+1:]
                rc1 = size.find(" ")
                name = size
                size = size[:rc1+1]
                
                rc1 = name.find(" ")
                name = name[rc1+1:]
                rc1 = name.find(" ")
                name = name[rc1+1:]
                rc1 = name.find(" ")
                name = name[rc1+1:]
                rc1 = name.find(" ")
                name = name[rc1+1:]
                rc1 = name.find(" ")
                name = name[rc1+1:]
                rc1 = name.find(" ")
                table_data.append([name, size, "#"])
            else:
                size = line
                rc1 = size.find(" ")
                size = size[rc1+1:]
                rc1 = size.find(" ")
                size = size[rc1+1:]
                rc1 = size.find(" ")
                size = size[rc1+1:]
                rc1 = size.find(" ")
                size = size[rc1+1:]
                rc1 = size.find(" ")
                name = size
                size = size[:rc1+1]
                
                rc1 = name.find(" ")
                name = name[rc1+1:]
                rc1 = name.find(" ")
                name = name[rc1+1:]
                rc1 = name.find(" ")
                name = name[rc1+1:]
                rc1 = name.find(" ")
                name = name[rc1+1:]
                rc1 = name.find(" ")
                name = name[rc1+1:]
                rc1 = name.find(" ")
                table_data.append([name, "#", size])
                table_data.append(["", "", ""])

    table_data.append(["not change", "493", "209"])
    table = AsciiTable(table_data)
    print table.table

#generateByLeters()
#generateReal()
#generateFake()

runTest()
gatherTar()
gatherResult()
