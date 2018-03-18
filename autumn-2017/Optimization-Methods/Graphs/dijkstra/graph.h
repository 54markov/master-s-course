#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <vector>

class AdjListNode
{
    public:
        AdjListNode(int d, int w);
        ~AdjListNode();

        int dest;
        int weight;
        AdjListNode *next;
};


class AdjList
{
    public:
        AdjList(AdjListNode *h);
        ~AdjList();
        AdjListNode *head; // Pointer to head node of list   
};
 
class Graph
{
    public:
        Graph(int v);
        ~Graph();

        void dijkstra(const int src);
        void addEdge(const int src, const int dest, const int weight);

        void printSolution(const std::vector<int> &dist,
                           const std::vector<int> &parent,
                           const int src);

        void printPath(const std::vector<int> &parent, int idx);

    private:
        int v_;
        std::vector<AdjList> list_;
};

#endif /* _GRAPH_H_ */
