all: unbounded_knapsack_problem bounded_knapsack_problem knapsack

unbounded_knapsack_problem: unbounded_knapsack_problem.cpp
	mkdir -p build
	g++ -g -Wall -o build/ukp unbounded_knapsack_problem.cpp -std=c++17

bounded_knapsack_problem: bounded_knapsack_problem.cpp
	mkdir -p build
	g++ -g -Wall -o build/bkp bounded_knapsack_problem.cpp -std=c++17

knapsack: knapsack.cpp
	mkdir -p build
	g++ -g -Wall -o build/ks knapsack.cpp -std=c++17

clean:
	rm -Rf build

