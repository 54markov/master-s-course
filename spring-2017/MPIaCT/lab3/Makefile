all: main

main: main.cpp element pipeline
	g++ -g -Wall -o main main.cpp Element.o PipelineStage.o -std=c++14

element: Element.cpp Element.h
	g++ -g -Wall -c Element.cpp -std=c++14

pipeline: PipelineStage.cpp PipelineStage.h
	g++ -g -Wall -c PipelineStage.cpp -std=c++14

clean:
	rm -f *.o main
