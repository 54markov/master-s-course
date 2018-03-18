all: main

main: main.cpp task manager worker tsqueue tscontainer 
	g++ -g -Wall -O3 -o main main.cpp task.o manager.o worker.o thread-safe-queue.o thread-safe-container.o -std=c++17 -lpthread

task: task.cpp task.h
	g++ -g -Wall -O3 -c task.cpp -std=c++17

manager: manager.cpp manager.h
	g++ -g -Wall -O3 -c manager.cpp -std=c++17

worker: worker.cpp worker.h
	g++ -g -Wall -O3 -c worker.cpp -std=c++17

tsqueue: thread-safe-queue.cpp thread-safe-queue.h
	g++ -g -Wall -O3 -c thread-safe-queue.cpp -std=c++17

tscontainer: thread-safe-container.cpp thread-safe-container.h
	g++ -g -Wall -O3 -c thread-safe-container.cpp -std=c++17

clean:
	rm -f main *.o
