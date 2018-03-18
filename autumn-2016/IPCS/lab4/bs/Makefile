all: bs zk rc4 rand erase

bs:
	g++ -g -O3 -o bs bs.cpp -std=c++14

zl:
	g++ -g -O3 -o zk zk.cpp -std=c++14


rc4:
	g++ -g -O3 -o rc4 rc4.cpp -std=c++14

rand:
	g++ -g -o random random.cpp -std=c++14

erase:
	g++ -g -o erase_image erase_image.cpp -std=c++14

clean:
	rm -f bs zk rc4 random erase_image rand_file mountains file *.bin
