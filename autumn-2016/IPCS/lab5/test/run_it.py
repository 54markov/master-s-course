#!/usr/bin/python
import commands
import os
from terminaltables import AsciiTable

pathToImage = "../images/"

imagesList = [ "PU24-1.bmp", "PU24-2.bmp", "PU24-3.bmp", "PU24-4.bmp",
               "PU24-5.bmp", "PU24-10.bmp", "PU24-11.bmp", "PU24-12.bmp", 
               "PU24-13.bmp" ]


############################ MAIN #############################################

os.system("rm -f statistic")
os.system("rm -f *.bmp")
os.system("rm -f *.tar.gz")

print "\n########################HAMM## ENCODING ##################################\n"

for image in imagesList:
    cmd = "./hamm1 -e " + pathToImage + image + " hamm_" + image + " this_is_a_test_secert_message_" + image
    os.system(cmd)

print "\n########################LSB_R# ENCODING ##################################\n"

for image in imagesList:
    cmd = "./lsb_r -e " + pathToImage + image + " lsb_r_" + image + " this_is_a_test_secert_message_" + image
    os.system(cmd)

print "\n########################LSB_R# DECODING ##################################\n"

for image in imagesList:
    cmd = "./lsb_r -d lsb_r_" + image
    os.system(cmd)

print "\n########################LSB_M# ENCODING ##################################\n"

for image in imagesList:
    cmd = "./lsb_m -e " + pathToImage + image + " lsb_m_" + image + " this_is_a_test_secert_message_" + image
    os.system(cmd)

print "\n########################LSB_M# DECODING ##################################\n"

for image in imagesList:
    cmd = "./lsb_m -d lsb_m_" + image
    os.system(cmd)

print "\n############################## CREATING TAR.GZ ############################\n"
for image in imagesList:
    cmd = "tar -zcvf lsb_m_" + image + ".tar.gz lsb_m_" + image
    os.system(cmd)
    cmd = "tar -zcvf lsb_r_" + image + ".tar.gz lsb_r_" + image
    os.system(cmd)
    cmd = "tar -zcvf hamm_" + image + ".tar.gz hamm_" + image
    os.system(cmd)

print "\n######################### CREATING ORIG TAR.GZ ############################\n"
for image in imagesList:
    cmd = "tar -zcvf orig_" + image + ".tar.gz " + pathToImage + image
    os.system(cmd)

os.system("ls -l | grep '.tar.gz' >> statistic")

table_data = []

table_data.append(['FILE', 'ORIGINAL', 'LSB_R', 'LSB_M', 'HAMMING'])

for image in imagesList:
    a = " "
    b = " "
    c = " "
    d = " "
    with open("statistic") as file:
        for line in file:
            if line.find(image) >= 0:
                if line.find("orig") >= 0:
                    a = line[30:-30]
                if line.find("lsb_r") >= 0:
                    b = line[30:-30]
                if line.find("lsb_m") >= 0:
                    c = line[30:-30]
                if line.find("hamm") >= 0:
                    d = line[30:-30]

    table_data.append([image, a[:5], b[:5], c[:5], d[:5]])

table = AsciiTable(table_data)
print table.table