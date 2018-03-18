all: main

main: ford-bellman
	mkdir -p build
	g++ -g -Wall -O2 -o build/ford-bellman main.cpp build/ford-bellman.o -std=c++17

ford-bellman:
	mkdir -p build
	g++ -g -Wall -c -o build/ford-bellman.o ford-bellman.cpp -std=c++17

clean:
	rm -Rf build

