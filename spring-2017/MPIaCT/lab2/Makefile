all: main

main: main.cpp functions classes
	g++ -g -Wall -o main main.cpp functions.o classes.o -std=c++14

functions: functions.cpp functions.h
	g++ -g -Wall -c functions.cpp -std=c++14

classes: classes.cpp classes.h
	g++ -g -Wall -c classes.cpp -std=c++14

clean:
	rm -f *.o main
