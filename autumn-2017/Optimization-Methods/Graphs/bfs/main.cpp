#include <iostream>

#include "bfs.h"

int main(int argc, char const *argv[])
{
    const auto source = 0;

    Graph graph(10);
    
    graph.setEdge(0, 1, 1);
    graph.setEdge(0, 3, 1);
    graph.setEdge(0, 4, 1);
    graph.setEdge(1, 4, 1);
    graph.setEdge(1, 2, 1);
    graph.setEdge(2, 7, 1);
    graph.setEdge(3, 2, 1);
    graph.setEdge(4, 3, 1);
    graph.setEdge(4, 2, 1);
    graph.setEdge(5, 6, 1);
    graph.setEdge(6, 7, 1);
    graph.setEdge(7, 8, 1);
    graph.setEdge(7, 9, 1);

    std::cout << "Graph:" << std::endl;

    std::cout << " 0-----1      " << std::endl;
    std::cout << " |\\   /|     " << std::endl;
    std::cout << " | \\4/ |      " << std::endl;
    std::cout << " | / \\ |     " << std::endl;
    std::cout << " |/   \\|     " << std::endl;
    std::cout << " 3-----2      " << std::endl;
    std::cout << " |     |      " << std::endl;
    std::cout << " 5--6--7----8 " << std::endl;
    std::cout << "       |      " << std::endl;
    std::cout << "       9      " << std::endl;

    std::cout << "Running Breadth-First Search:" << std::endl;
    std::cout << "Source vertex is " << source << std::endl;

    graph.bfs(source);

    return 0;
}
