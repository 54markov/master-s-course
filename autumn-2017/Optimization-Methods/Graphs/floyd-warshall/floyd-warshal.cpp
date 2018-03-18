/*
 * C Program for Floyd Warshall Algorithm
 * http://www.geeksforgeeks.org/dynamic-programming-set-16-floyd-warshall-algorithm/
 */
#include <iostream>
#include <algorithm>

#include "floyd-warshal.h"


FloydWarshal::FloydWarshal() { }
FloydWarshal::~FloydWarshal() { }

/* 
 * A utility function to print solution
 */
void FloydWarshal::printSolution(std::vector<std::vector<int>> &v)
{
    std::cout << "Following matrix shows the shortest distances ";
    std::cout << "between every pair of vertices" << std::endl;

    for (auto iter : v)
    {
        for(auto distance : iter)
        {
            if (distance == INF)
                std::cout << "INF" << "\t";
            else
                std::cout << distance << "\t";
        }
        std::cout << std::endl;
    }
}
 
/*
 * Solves the all-pairs shortest path problem using Floyd Warshall algorithm
 */
void FloydWarshal::floydWarshall()
{
    /*
     * dist[][] will be the output matrix that will
     * finally have the shortest 
     * distances between every pair of vertices
     */
    std::vector<std::vector<int>> dist(this->vertices_,
        std::vector<int>(this->vertices_, 0));
 
    /* 
     * Initialize the solution matrix same as input graph matrix.
     * Or we can say the initial values of shortest distances are based
     * on shortest paths considering no intermediate vertex.
     */
    for (auto i = 0; i < this->vertices_; ++i)
    {
        for (auto j = 0; j < this->vertices_; ++j)
        {
            dist[i][j] = this->graph_[i][j];
        }
    }
 
    /* 
     * Add all vertices one by one to the set of intermediate vertices.
     *  ---> Before start of a iteration, we have shortest distances between all
     * pairs of vertices such that the shortest distances consider only the
     * vertices in set {0, 1, 2, .. k-1} as intermediate vertices.
     * ----> After the end of a iteration, vertex no. k is added to the set of
     * intermediate vertices and the set becomes {0, 1, 2, .. k}
     */
    for (auto k = 0; k < this->vertices_; ++k)
    {
        /*
         * Pick all vertices as source one by one
         */
        for (auto i = 0; i < this->vertices_; ++i)
        {
            /* 
             * Pick all vertices as destination for the above picked source
             */
            for (auto j = 0; j < this->vertices_; ++j)
            {
                /*
                 * If vertex k is on the shortest path from
                 * i to j, then update the value of dist[i][j]
                 */
                if (dist[i][k] + dist[k][j] < dist[i][j])
                {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
 
    /*
     * Print the shortest distance matrix
     */
    this->printSolution(dist);
}

void FloydWarshal::createTestGraph(int vertices)
{
    this->vertices_ = vertices;

    this->graph_.push_back(std::vector<int>( { 0,   5,   INF, 10  } ));
    this->graph_.push_back(std::vector<int>( { INF, 0,   3,   INF } ));
    this->graph_.push_back(std::vector<int>( { INF, INF, 0,   1   } ));
    this->graph_.push_back(std::vector<int>( { INF, INF, INF, 0   } ));
}
