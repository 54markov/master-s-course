all: main generator

main:
	mkdir -p build
	g++ -g -Wall -O2 -o build/max-flow-problem main.cpp -std=c++17

generator:
	mkdir -p build
	g++ -g -Wall -O2 -o build/generator generator.cpp -std=c++17

clean:
	rm -Rf build
