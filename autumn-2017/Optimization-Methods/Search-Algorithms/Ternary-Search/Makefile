all: ternary_search

ternary_search: ternary_search.cpp
	g++ -g -Wall -O3 -o ts_gs ternary_search.cpp -std=c++17

ternary_search_simple: ternary_search.cpp
	g++ -g -Wall -O3 -o ts_gs_simple ternary_search.cpp -std=c++17 -D SIMPLE

clean:
	rm -f ts_gs ts_gs_simple
