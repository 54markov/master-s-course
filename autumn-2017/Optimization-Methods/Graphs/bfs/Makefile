all: main generator test

main: bfs
	mkdir -p build
	g++ -g -Wall -O2 -o build/main main.cpp build/bfs.o -std=c++17

bfs:
	mkdir -p build
	g++ -g -Wall -O2 -c -o build/bfs.o bfs.cpp -std=c++17

test: bfs
	mkdir -p build
	g++ -g -Wall -O2 -o build/test test.cpp build/bfs.o -std=c++17

generator:
	mkdir -p build
	g++ -g -Wall -O2 -o build/generator generator.cpp -std=c++17	

clean:
	rm -Rf build
