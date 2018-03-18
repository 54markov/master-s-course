#include <iostream>
#include <chrono>
#include <vector>
#include <queue>

#include "bfs.h"

Graph::Graph(const int nvertices)
{
    this->vertices_ = nvertices;
    this->edges_    = std::vector<std::vector<Node_t>>(nvertices);
}

Graph::~Graph() { }

void Graph::setEdge(int src, int dst, int w)
{
    this->edges_[src].push_back( { dst, w } );
    this->edges_[dst].push_back( { src, w } );
}

void Graph::bfs(const int src)
{
    auto path = 0;
    std::queue<int> q;
    std::vector<bool> visited(this->vertices_, false);

    visited[src] = true;
    q.push(src);

    std::chrono::high_resolution_clock::time_point t1 =  std::chrono::high_resolution_clock::now();

    while (!q.empty())
    {
        auto v = q.front();
        q.pop();
        for (int i = 0; i < static_cast<int>(this->edges_[v].size()); i++)
        {
            if (this->edges_[v][i].weight > 0 && !visited[this->edges_[v][i].dst])
            {
                q.push(this->edges_[v][i].dst);
                visited[this->edges_[v][i].dst] = true;
                path += this->edges_[v][i].weight;

                /*
                 * Print path for the last vertex
                 */
                if (this->vertices_ == this->edges_[v][i].dst + 1)
                {
                    std::cout << "Vertex: " << this->edges_[v][i].dst + 1 << std::endl;
                    std::cout << "Path  : " << path << std::endl;
                }
            }
        }
    }

    std::chrono::high_resolution_clock::time_point t2 =  std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Elapsed time is: " << time_span.count() << " seconds" << std::endl;
}
