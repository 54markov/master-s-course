#include <limits> 
#include <iostream>

#include "ford-bellman.h"

FordBellman::FordBellman(int v, int e) : v_(v), e_(e) { }
FordBellman::~FordBellman() { }

/*
 * The main function that finds shortest distances from src to
 * all other vertices using Bellman-Ford algorithm.  The function
 * also detects negative weight cycle
 */
void FordBellman::bellmanFord(int src)
{
    /*
     * Step 1: Initialize distances from src
     * to all other vertices as INFINITE
     */
    std::vector<int> dist(this->v_, std::numeric_limits<int>::max());

    dist[src] = 0;
 
    /* 
     * Step 2: Relax all edges |V| - 1 times. A simple shortest 
     * path from src to any other vertex can have at-most |V| - 1 edges
     */
    for (auto i = 1; i < this->v_; i++)
    {
        for (auto j = 0; j < this->e_; j++)
        {
            auto u      = this->graph_[j].src;
            auto v      = this->graph_[j].dest;
            auto weight = this->graph_[j].weight;

            if ((dist[u] != std::numeric_limits<int>::max()) &&
                ((dist[u] + weight) < dist[v]))
            {
                dist[v] = dist[u] + weight;
            }
        }
    }
 
    /*
     * Step 3: check for negative-weight cycles.
     * The above step guarantees shortest distances if graph doesn't contain
     * negative weight cycle.
     * If we get a shorter path, then there is a cycle.
     */
    for (auto i = 0; i < this->e_; i++)
    {
        auto u      = this->graph_[i].src;
        auto v      = this->graph_[i].dest;
        auto weight = this->graph_[i].weight;

        if ((dist[u] != std::numeric_limits<int>::max()) &&
            ((dist[u] + weight) < dist[v]))
        {
            std::cout << "Graph contains negative weight cycle" << std::endl;
        }
    }
 
    this->printSolution(dist);
}

/*
 * A utility function used to print the solution
 */
void FordBellman::printSolution(std::vector<int> &v)
{
    std::cout << "Vertex\tDistance from Source" << std::endl;

    for (auto i = 0; i < static_cast<int>(v.size()); ++i)
    {
        std::cout << i << "\t" << v[i] << std::endl;
    }
}

void FordBellman::createTestGraph()
{
    this->graph_.push_back({ .src = 0, .dest = 1, .weight = 25 });
    this->graph_.push_back({ .src = 0, .dest = 2, .weight = 15 });
    this->graph_.push_back({ .src = 0, .dest = 3, .weight = 7 });
    this->graph_.push_back({ .src = 0, .dest = 4, .weight = 2 });

    this->graph_.push_back({ .src = 1, .dest = 0, .weight = 25 });
    this->graph_.push_back({ .src = 1, .dest = 2, .weight = 6 });

    this->graph_.push_back({ .src = 2, .dest = 0, .weight = 15 });
    this->graph_.push_back({ .src = 2, .dest = 1, .weight = 6 });
    this->graph_.push_back({ .src = 2, .dest = 3, .weight = 4 });

    this->graph_.push_back({ .src = 3, .dest = 0, .weight = 7 });
    this->graph_.push_back({ .src = 3, .dest = 2, .weight = 4 });
    this->graph_.push_back({ .src = 3, .dest = 4, .weight = 3 });

    this->graph_.push_back({ .src = 4, .dest = 0, .weight = 2 });
    this->graph_.push_back({ .src = 4, .dest = 3, .weight = 3 });
}
