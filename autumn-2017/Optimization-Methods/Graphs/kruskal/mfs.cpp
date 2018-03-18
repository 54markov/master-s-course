#include "mfs.h"
#include "heap.h"
#include "graph.h"

#include <string>
#include <iostream>

MFSet::MFSet(const int nelems)
{
    this->nelems_ = nelems;
    this->nsets_  = 0;

    this->sets_  = new set_t[nelems];
    if (!this->sets_)
        throw std::string("Can't allocate memory");

    this->elems_ = new elem_t[nelems];
    if (!this->elems_)
        throw std::string("Can't allocate memory");

    for (auto i = 0; i < nelems; ++i)
    {
        this->sets_[i].size  = 0;
        this->sets_[i].first = -1;
        this->elems_[i].set  = -1;
        this->elems_[i].next = -1;
    }
}

MFSet::~MFSet()
{
    delete[] this->sets_;
    delete[] this->elems_;
}

void MFSet::makeSet(int elem)
{
    this->sets_[this->nsets_].size  = 1;
    this->sets_[this->nsets_].first = elem;
    this->elems_[elem].set          = this->nsets_;
    this->elems_[elem].next         = -1;
    this->nsets_++;
}

int MFSet::findSet(const int elem)
{
    return this->elems_[elem].set;
}

void MFSet::unionSet(int elem1, int elem2)
{
    auto set1 = this->findSet(elem1);
    auto set2 = this->findSet(elem2);

    if (this->sets_[set1].size < this->sets_[set2].size)
    {
        auto temp = set1;
        set1 = set2;
        set2 = temp;
    }

    /*
     * S1 > S2; Merge elems of S2 to S1
     */
    auto i = this->sets_[set2].first;
    while (this->elems_[i].next != -1)
    {
        this->elems_[i].set = set1;
        i = this->elems_[i].next;
    }

    /* 
     * Add elems of S1 to the end of S2
     */
    this->elems_[i].set     = set1;
    this->elems_[i].next    = this->sets_[set1].first;
    this->sets_[set1].first = this->sets_[set2].first;
    this->sets_[set1].size += this->sets_[set2].size;

    /* 
     * Remove S2
     */
    this->sets_[set2].size  = 0;
    this->sets_[set2].first = -1;
    this->nsets_--;
}

int MFSet::searchKruskal(Graph *g, Graph *mst)
{
    /*
     * Insert edges in heap
     */
    Heap pq(g->getVerctices() * g->getVerctices());
    
    const auto n = g->getVerctices();
    auto mstlen = 0;
    
    for (auto i = 0; i < n; ++i)
    {
        this->makeSet(i);
    }

    /*
     * For all edges (adj. matrix)
     */
    for (auto i = 0; i < n; ++i)
    {
        for (auto j = i + 1; j < n; ++j)
        {
            auto w = g->getEdge(i + 1, j + 1);
            if (w > 0)
            {
                struct heapvalue edge = { .i = i, .j = j };
                pq.insert(w, edge);
            }
        }
    }

    std::cout << "Minimum spanning tree edges:" << std::endl;
    for (auto i = 0; i < n - 1; )
    {
        auto item = pq.removeMin();
        auto s1 = this->findSet(item.value.i);
        auto s2 = this->findSet(item.value.j);
        if (s1 != s2)
        {
            char u = 'A' + item.value.i;
            char v = 'A' + item.value.j;
            std::cout << u << " - " << v << std::endl;

            this->unionSet(item.value.i, item.value.j);
            mstlen += item.priority;
            mst->setEdge(item.value.i + 1, item.value.j + 1, item.priority);
            i++;
        }
    }
    std::cout << std::endl;
    std::cout << "Minimum spanning tree weight: " << mstlen << std::endl;

    return mstlen;
}
