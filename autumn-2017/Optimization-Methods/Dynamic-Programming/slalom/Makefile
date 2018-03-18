all: slalom slalom_stdio

slalom: slalom.cpp
	g++ -g -Wall -o slalom_stdio slalom_stdio.cpp -std=c++17

slalom_stdio: slalom_stdio.cpp
	g++ -g -Wall -o slalom_stdio slalom_stdio.cpp -std=c++17

clean:
	rm -f slalom slalom_stdio slalom.out
