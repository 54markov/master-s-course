all: main debug

main: task calendar statistic main.cpp
	mkdir -p build
	g++ -g -Wall -O2 -o build/main build/task.o build/statistic.o build/calendar.o main.cpp -std=c++17

task: task.cpp task.h
	mkdir -p build
	g++ -g -Wall -O2 -c -o build/task.o task.cpp -std=c++17

statistic: statistic.cpp statistic.h
	mkdir -p build
	g++ -g -Wall -O2 -c -o build/statistic.o statistic.cpp -std=c++17

calendar: calendar.cpp calendar.h
	mkdir -p build
	g++ -g -Wall -O2 -c -o build/calendar.o calendar.cpp -std=c++17

debug: taskd calendard statisticd main.cpp
	mkdir -p build
	g++ -g -Wall -O2 -o build/debug build/task.o build/statistic.o build/calendar.o main.cpp -std=c++17 -D DEBUG

taskd: task.cpp task.h
	mkdir -p build
	g++ -g -Wall -O2 -c -o build/task.o task.cpp -std=c++17 -D DEBUG

statisticd: statistic.cpp statistic.h
	mkdir -p build
	g++ -g -Wall -O2 -c -o build/statistic.o statistic.cpp -std=c++17 -D DEBUG

calendard: calendar.cpp calendar.h
	mkdir -p build
	g++ -g -Wall -O2 -c -o build/calendar.o calendar.cpp -std=c++17 -D DEBUG

clean:
	rm -Rf build