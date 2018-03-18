all: main

main: floyd-warshal
	mkdir -p build
	g++ -g -Wall -O2 -o build/main main.cpp build/floyd-warshal.o -std=c++17

floyd-warshal:
	mkdir -p build
	g++ -g -Wall -c -o build/floyd-warshal.o floyd-warshal.cpp -std=c++17

clean:
	rm -Rf build
