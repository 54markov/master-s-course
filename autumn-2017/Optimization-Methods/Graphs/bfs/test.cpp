#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <chrono>

#include "bfs.h"

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

        auto vertecies = /*std::sqrt*/(std::stoi(line));

        Graph graph(vertecies);

        while (getline(file, line))
        {
            int src, dst, weight;
            std::stringstream ss;
            ss << line;
            ss >> src;
            ss >> dst;
            ss >> weight;

            //std::cout << "Added edge: " << src << " " << dst << " " << weight << std::endl;
            graph.setEdge(src, dst, weight);
        }

        file.close();

        graph.bfs(source);

        std::cout << "Running Breadth-First Search:" << std::endl;
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
