/*
 * Bellman-Ford's single source shortest path algorithm
 * http://www.geeksforgeeks.org/dynamic-programming-set-23-bellman-ford-algorithm/
 */

#ifndef _FORD_BELLMAN_H_
#define _FORD_BELLMAN_H_

#include <vector>

/*
 * The structure to represent a weighted edge in graph
 */
struct Edge
{
    int src;
    int dest;
    int weight;
};

class FordBellman
{
    public:
        FordBellman(int v, int e);
        ~FordBellman();

        void createTestGraph();
        void bellmanFord(int src);
        void printSolution(std::vector<int> &v);

    private:
        int v_; // vertices
        int e_; // edges
        /*
         * The vector of edge is represent a connected, 
         * directed and weighted graph
         */
        std::vector<struct Edge> graph_;
};

#endif /* _FORD_BELLMAN_H_ */
