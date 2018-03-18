/*
 * Dijkstra's shortest path algorithm for adjacency list representation of graph
 * http://www.geeksforgeeks.org/greedy-algorithms-set-7-dijkstras-algorithm-for-adjacency-list-representation/
 */

#include "graph.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <chrono>

int main(int argc, char const *argv[])
{
    const auto source = 0;

    std::ifstream file("build/graph.txt");

    try
    {
        if (!file.is_open())
            throw std::string("Can't read from file");

        std::string line;
        getline(file, line);

        std::cout << "Graph size is : " << line << std::endl;

        Graph graph(std::stoi(line));

        while (getline(file, line))
        {
            int src, dst, weight;
            std::stringstream ss;
            ss << line;
            ss >> src;
            ss >> dst;
            ss >> weight;
            graph.addEdge(src, dst, weight);
/*
            std::cout << "Added edge: " << src << " " << dst << " " << weight << std::endl;
*/
        }

        file.close();

        graph.dijkstra(source);

        std::cout << "Running Dijkstra algorithm" << std::endl;
        std::cout << "Graph:" << std::endl;
        std::cout << " (0)---1---(1)---1---(n)  " << std::endl;
        std::cout << " |          |         |  " << std::endl;
        std::cout << " |          |         |  " << std::endl;
        std::cout << " |          |         |  " << std::endl;
        std::cout << " 1          1         1   " << std::endl;
        std::cout << " |          |         |  " << std::endl;
        std::cout << " |          |         |  " << std::endl;
        std::cout << " |          |         |  " << std::endl;
        std::cout << " (.)---1---(.)---1---(n)  " << std::endl;
        std::cout << "Source vertex is '" <<  source << "'" << std::endl;
    }
    catch (std::string err)
    {
        std::cerr << err << std::endl;
        return -1;
    }
 
    return 0;
}
