all: main

main: graph mfs heap
	mkdir -p build
	g++ -g -Wall -o build/kruskal main.cpp build/graph.o build/mfs.o build/heap.o -std=c++17

mfs:
	mkdir -p build
	g++ -g -Wall -c -o build/mfs.o mfs.cpp -std=c++17

graph:
	mkdir -p build
	g++ -g -Wall -c -o build/graph.o graph.cpp -std=c++17

heap:
	mkdir -p build
	g++ -g -Wall -c -o build/heap.o heap.cpp -std=c++17

clean:
	rm -Rf build
