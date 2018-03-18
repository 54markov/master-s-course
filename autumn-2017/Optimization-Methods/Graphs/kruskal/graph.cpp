#include "graph.h"

#include <iostream>
#include <queue>

Graph::Graph(const int nvertices)
{
    this->nvertices_ = nvertices;

    this->visited_ = new int[nvertices];

    this->m_ = new int[nvertices * nvertices];

    this->clear(); // Опционально, O(n^2)
}

Graph::~Graph()
{
    delete[] this->visited_;
    delete[] this->m_;
}

void Graph::clear()
{
    for (auto i = 0; i < this->nvertices_; ++i)
    {
        this->visited_[i] = 0;
        for (auto j = 0; j < this->nvertices_; j++)
        {
            this->m_[i * this->nvertices_ + j] = 0;
        }
    }
}

void Graph::setEdge(const int i, const int j, int w)
{
    this->m_[(i - 1) * this->nvertices_ + j - 1] = w;
    this->m_[(j - 1) * this->nvertices_ + i - 1] = w;
}

int Graph::getEdge(const int i, const int j)
{
    return this->m_[(i - 1) * this->nvertices_ + j - 1];
}

int Graph::getVerctices()
{
    return this->nvertices_;
}
