#ifndef _GRAPH_BFS_H_
#define _GRAPH_BFS_H_

#include <vector>

typedef struct
{
    int dst; 
    int weight;
} Node_t;

class Graph
{
    public:
        Graph(const int nvertices);
        ~Graph();

        void setEdge(int src, int dst, int w);
        void bfs(const int src);

    private:
        int vertices_;
        std::vector<std::vector<Node_t>> edges_;
};

#endif /* _GRAPH_BFS_H_ */
