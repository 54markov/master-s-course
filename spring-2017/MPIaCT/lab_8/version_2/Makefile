all: main

main: plot point
	g++ -g -Wall -O3 -o main main.cpp plot.o point.o -std=c++17 -lm -lpng

point:
	g++ -g -Wall -O3 -c point.cpp -std=c++17 -lm -lpng

plot:
	g++ -g -Wall -O3 -c plot.cpp -std=c++17 -lm -lpng

clean:
	rm -f main *.png *.o
