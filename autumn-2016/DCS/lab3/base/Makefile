all: main

main: storage.o timer.o customer.o loader.o
	g++ -g -o main main.cpp storage.o timer.o customer.o loader.o -std=c++14 -pthread

storage.o: storage.cpp storage.h
	g++ -g -c storage.cpp -std=c++14

timer.o: timer.cpp timer.h
	g++ -g -c timer.cpp -std=c++14

loader.o: loader.cpp loader.h
	g++ -g -c loader.cpp -std=c++14

customer.o: customer.cpp customer.h
	g++ -g -c customer.cpp -std=c++14

clean:
	rm -f *.o main
