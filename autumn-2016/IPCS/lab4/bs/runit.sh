#!/bin/sh

rm -f rc4.bin zk.bin file rand_file

./rc4 -k fedcba98765432100123456789abcdef -n 134217728 >> rc4.bin

./zk -k fedcba98765432100123456789abcdef -n 134217728 >> zk.bin

./random

./erase_image >> mountains

echo "\n-----------------------------" >> file
echo "./bs -f rc4.bin -w 16 -b 16 -u 5" >> file
./bs -f rc4.bin -w 16 -b 16 -u 5 >> file


echo "\n-----------------------------" >> file
echo "./bs -f zk.bin -w 16 -b 16 -u 5" >> file
./bs -f zk.bin -w 16 -b 16 -u 5 >> file


echo "\n-----------------------------" >> file
echo "./bs -f zk.bin -w 16 -b 16 -u 5 -n 256" >> file
./bs -f zk.bin -w 16 -b 16 -u 5 -n 256 >> file


echo "\n-----------------------------" >> file
echo "./bs -f rc4.bin" >> file
./bs -f rc4.bin >> file


echo "\n-----------------------------" >> file
echo "./bs -f zk.bin" >> file
./bs -f zk.bin >> file

echo "\n-----------------------------" >> file
echo "./bs -f zk.bin -n 128" >> file
./bs -f zk.bin -n 128 >> file

echo "\n-----------------------------" >> file
echo "./bs -f rc4.bin -w 16 -b 16 -u 10000" >> file
./bs -f rc4.bin -w 16 -b 16 -u 10000 >> file

echo "\n-----------------------------" >> file
echo "./bs -f zk.bin -w 16 -b 16 -u 10000" >> file
./bs -f zk.bin -w 16 -b 16 -u 10000 >> file



echo "\n-----------------------------" >> file
echo "./bs -f mountains.jpg" >> file
./bs -f mountains.jpg >> file

echo "\n-----------------------------" >> file
echo "./bs -f mountains.jpg -n 4096" >> file
./bs -f mountains.jpg -n 4096 >> file

echo "\n-----------------------------" >> file
echo "./bs -f mountains.jpg -w 16 -b 16 -u 10000" >> file
./bs -f mountains.jpg -w 16 -b 16 -u 10000 >> file

echo "\n-----------------------------" >> file
echo "./bs -f rand_file -w 32 -b 8" >> file
./bs -f rand_file -w 32 -b 8 >> file

echo "\n-----------------------------" >> file
echo "./bs -f rand_file" >> file
./bs -f rand_file >> file

echo "\n-----------------------------" >> file
echo "./bs -f rand_file -n 256" >> file
./bs -f rand_file -n 256  >> file

echo "\n-----------------------------" >> file
echo "./bs -f mountains(erased 54) -n 256" >> file
./bs -f mountains -n 256 >> file


./parse.py file