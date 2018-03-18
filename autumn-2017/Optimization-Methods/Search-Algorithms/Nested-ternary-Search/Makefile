all: nested_ternary_search

nested_ternary_search: nested_ternary_search.cpp
	g++ -g -Wall -O3 -o ns_ts nested_ternary_search.cpp -std=c++17 -lm

test: test.cpp
	g++ -g -Wall -O3 -o test test.cpp -std=c++17 -lm

clean:
	rm -f ns_ts test
