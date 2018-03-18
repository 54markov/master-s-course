all: task3

task3: main book_stack code_number io_file
	g++ -g -o main main.o code_number.o book_stack.o io_file.o -std=c++14

main: main.cpp
	g++ -g -c main.cpp -std=c++14

code_number: code_number.cpp code_number.h
	g++ -g -c code_number.cpp -std=c++14

book_stack: book_stack.cpp book_stack.h
	g++ -g -c book_stack.cpp -std=c++14

io_file: io_file.cpp io_file.h
	g++ -g -c io_file.cpp -std=c++14

clean:
	rm -f *.o main compressed_file
