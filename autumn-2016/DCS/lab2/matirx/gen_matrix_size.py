#!/usr/bin/python
import commands
import os

count = 16
f = open('workfile', 'w')

while count <= 4096:
	string = str(count)
	f.write(string)
	f.write("\n")

	count += 16

f.close()