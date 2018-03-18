/*
 * Dijkstra's shortest path algorithm for adjacency list representation of graph
 * http://www.geeksforgeeks.org/greedy-algorithms-set-7-dijkstras-algorithm-for-adjacency-list-representation/
 */

#include "graph.h"

#include <iostream>

int main(int argc, char const *argv[])
{
    const auto source = 0;

    Graph graph(5);

    graph.addEdge(0, 1, 25);
    graph.addEdge(0, 2, 15);
    graph.addEdge(0, 3, 7);
    graph.addEdge(0, 4, 2);
    graph.addEdge(1, 2, 6);
    graph.addEdge(2, 3, 4);
    graph.addEdge(3, 4, 3);

    std::cout << "Running Dijkstra algorithm" << std::endl;
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

    graph.dijkstra(source);
 
    return 0;
}
