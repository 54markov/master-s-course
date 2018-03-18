#include <iostream>

#include "floyd-warshal.h"

#include <iostream>

/*
 * The Floyd Warshall Algorithm is for solving the All Pairs Shortest Path problem.
 * The problem is to find shortest distances between every pair of vertices
 * in a given edge weighted directed Graph.
 */
int main(int argc, char const *argv[])
{
    FloydWarshal graph;

    graph.createTestGraph(4);

    std::cout << "Running the Floyd Warshall Algorithm" << std::endl;
    std::cout << "Graph: " << std::endl;
    std::cout << "        10        " << std::endl;
    std::cout << "  (0)------->(3)  " << std::endl;
    std::cout << "   |         /|\\ " << std::endl;
    std::cout << " 5 |          |   " << std::endl;
    std::cout << "   |          | 1 " << std::endl;
    std::cout << "  \\|/         |   " << std::endl;
    std::cout << "  (1)------->(2)  " << std::endl;
    std::cout << "        3         " << std::endl;

    graph.floydWarshall();

    return 0;
}
