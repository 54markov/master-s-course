#include "heap.h"

#include <string>

Heap::Heap(int maxsize)
{
    this->maxsize_ = maxsize;
    this->nnodes_  = 0;
    
    /*
     * Heap nodes [0, 1, ..., maxsize]
     */
    this->nodes_ = new struct heapitem[maxsize + 1];
    if (!this->nodes_) {
        throw std::string("Can't create heap");
    }
}

Heap::~Heap()
{
    delete[] this->nodes_;
}

void Heap::swap(struct heapitem *a, struct heapitem *b)
{
    auto temp = *a;
    *a = *b;
    *b = temp;
}

int Heap::insert(int priority, struct heapvalue value)
{
    if (this->nnodes_ >= this->maxsize_)
        throw std::string("heap overflow");

    this->nnodes_++;
    this->nodes_[this->nnodes_].priority  = priority;
    this->nodes_[this->nnodes_].value.i   = value.i;
    this->nodes_[this->nnodes_].value.j   = value.j;
    
    /*
     * HeapifyUp
    */
    for (auto i = this->nnodes_;
        i > 1 && this->nodes_[i].priority < this->nodes_[i / 2].priority;
        i = i / 2)
    {
        this->swap(&this->nodes_[i], &this->nodes_[i / 2]);
    }

    return 0;
}

void Heap::heapify(int index)
{
    while (true)
    {
        auto left  = 2 * index;
        auto right = 2 * index + 1;
        /*
         * Find smalest priority: A[index], A[left] and A[right]
         */
        auto smalest = index;
        if (left <= this->nnodes_ &&
            this->nodes_[left].priority < this->nodes_[index].priority)
        {
            smalest = left;
        }

        if (right <= this->nnodes_ &&
            this->nodes_[right].priority < this->nodes_[smalest].priority)
        {
            smalest = right;
        }

        if (smalest == index)
            break;

        this->swap(&this->nodes_[index], &this->nodes_[smalest]);
        index = smalest;
    }
}

struct heapitem Heap::removeMin()
{
    if (this->nnodes_ == 0)
    {
        struct heapitem empty;
        return empty;
    }

    auto minitem = this->nodes_[1];

    this->nodes_[1] = this->nodes_[this->nnodes_];
    this->nnodes_--;

    this->heapify(1);

    return minitem;
}
