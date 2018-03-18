#include "ford-bellman.h"

#include <iostream>

int main(int argc, char const *argv[])
{
	const auto source = 0;
    const auto v = 5;      // Number of vertices in graph
    const auto e = 2 * 7;  // Number of edges in graph

    FordBellman graph(v, e);

    graph.createTestGraph();

    std::cout << "Running Ford-Bellman algorithm" << std::endl;
    std::cout << "Graph:" << std::endl;
    std::cout << " (1)---25---(0)---2---(4)  " << std::endl;
    std::cout << " |           |\\        |  " << std::endl;
    std::cout << " |           | \\       |  " << std::endl;
    std::cout << " |           |  \\      |  " << std::endl;
    std::cout << " |           |   \\     |  " << std::endl;
    std::cout << " 6           15   7    3   " << std::endl;
    std::cout << " |           |     \\   |  " << std::endl;
    std::cout << " |           |      \\  |  " << std::endl;
    std::cout << " |           |       \\ |  " << std::endl;
    std::cout << " +----------(2)---4---(3)  " << std::endl;
    std::cout << "Source vertex is '" <<  source << "'" << std::endl;

    graph.bellmanFord(source);

    return 0;
}
