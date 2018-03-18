all: main generator test

main: graph heap
	mkdir -p build
	g++ -g -Wall -O2 -o build/dijkstra main.cpp build/graph.o build/heap.o -std=c++17

test: graph heap
	mkdir -p build
	g++ -g -Wall -O2 -o build/test test.cpp build/graph.o build/heap.o -std=c++17

graph:
	mkdir -p build
	g++ -g -Wall -O2 -c -o build/graph.o graph.cpp -std=c++17

heap:
	mkdir -p build
	g++ -g -Wall -O2 -c -o build/heap.o heap.cpp -std=c++17

generator:
	mkdir -p build
	g++ -g -Wall -O2 -o build/generator generator.cpp -std=c++17	


clean:
	rm -Rf build
