#include "heap.h"
#include <iostream>

MinHeapNode::MinHeapNode(const int _v, const int _dist) : v(_v), dist(_dist) { }
MinHeapNode::~MinHeapNode() { }

MinHeap::MinHeap(const int capacity)
{
    this->pos      = new int [capacity];
    this->counter  = capacity;
    this->capacity = capacity;
    this->heap     = new MinHeapNode* [capacity];
}

MinHeap::~MinHeap()
{
    delete[] this->pos;
    delete[] this->heap;
}

/*
 * Method to decreasy dist value of a given vertex v.
 * This Method uses pos[] of min heap to get the current index
 * of node in min heap
 */
void MinHeap::decreaseKey(const int v, const int dist)
{
    /*
     * Get the index of v in  heap array
     */
    auto i = this->pos[v];
 
    /*
     * Get the node and update its dist value
     */
    this->heap[i]->dist = dist;
 
    /*
     * Travel up while the complete tree is not hepified.
     * This is a O(Logn) loop
     */
    while (i && this->heap[i]->dist < this->heap[(i - 1) / 2]->dist)
    {
        /*
         * Swap this node with its parent
         */
        this->pos[this->heap[i]->v]           = (i - 1) / 2;
        this->pos[this->heap[(i - 1) / 2]->v] = i;

        this->swap(*this->heap[i], *this->heap[(i - 1) / 2]);
 
        /*
         * Move to parent index
         */
        i = (i - 1) / 2;
    }
}

/*
 * Standard method to extract minimum node from heap
 */
MinHeapNode *MinHeap::removeMin()
{
    if (this->isEmpty())
    {
        return nullptr;
    }

    // Store the root node
    auto *root = this->heap[0];

    // Replace root node with last node
    auto *lastNode = this->heap[this->counter - 1];
    this->heap[0]  = lastNode;

    // Update position of last node
    this->pos[root->v]     = this->counter - 1;
    this->pos[lastNode->v] = 0;

    // Reduce heap size and heapify root
    --this->counter;

    this->heapify(0);
 
    return root;
}

/*
 * A standard method to heapify at given idx
 * This method also updates position of nodes when they are swapped.
 * Position is needed for decreaseKey()
 */
void MinHeap::heapify(const int idx)
{
    auto smallest = idx;
    auto left     = 2 * idx + 1;
    auto right    = 2 * idx + 2;
 
    if (left < this->counter &&
        this->heap[left]->dist < this->heap[smallest]->dist)
    {
        smallest = left;
    }
 
    if (right < this->counter &&
        this->heap[right]->dist < this->heap[smallest]->dist)
    {
        smallest = right;
    }
 
    if (smallest != idx)
    {
        // The nodes to be swapped in min heap
        MinHeapNode *smallestNode = this->heap[smallest];
        MinHeapNode *idxNode      = this->heap[idx];
 
        // Swap positions
        this->pos[smallestNode->v] = idx;
        this->pos[idxNode->v]      = smallest;
 
        // Swap nodes
        this->swap(*this->heap[smallest], *this->heap[idx]);
 
        this->heapify(smallest);
    }
}

/*
 * A utility method to swap two nodes of min heap (needed for min heapify)
 */
void MinHeap::swap(MinHeapNode &a, MinHeapNode &b)
{
    MinHeapNode tmp = a;
    a = b;
    b = tmp;
}

/*
 * A utility method to check if the given minHeap is empty or not
 */
int MinHeap::isEmpty()
{
    return this->counter == 0;
}

/*
 * A utility method to check if a given vertex 'v' is in min heap or not
 */
bool MinHeap::isInHeap(const int v)
{
    if (this->pos[v] < this->counter)
    {
        return true;
    }
    return false;
}
