all: main

main: main.cpp field
	g++ -g -Wall -o main main.cpp Field.o -std=c++11

field: Field.cpp Field.h
	g++ -Wall -c Field.cpp -std=c++11

clean:
	rm -f main *.o