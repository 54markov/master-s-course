all: main

main: globalVariables.o keyBoardReader.o ticTacToySimulate.o
	g++ -g -o main main.cpp globalVariables.o keyBoardReader.o ticTacToySimulate.o -std=c++14 -pthread

globalVariables.o: globalVariables.cpp globalVariables.h
	g++ -g -c globalVariables.cpp -std=c++14

keyBoardReader.o: keyBoardReader.cpp keyBoardReader.h
	g++ -g -c keyBoardReader.cpp -std=c++14

ticTacToySimulate.o: ticTacToySimulate.cpp ticTacToySimulate.h
	g++ -g -c ticTacToySimulate.cpp -std=c++14

clean:
	rm -f *.o main
