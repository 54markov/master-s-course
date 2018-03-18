#include <algorithm>
#include <iostream>
#include <limits>
#include <chrono>

#include "heap.h"
#include "graph.h"


AdjListNode::AdjListNode(int d, int w) : dest(d), weight(w), next(nullptr) { }
AdjListNode::~AdjListNode() { }

AdjList::AdjList(AdjListNode *h = nullptr) : head(h) { }
AdjList::~AdjList() { }

Graph::Graph(const int v)
{
    this->v_ = v;
 
    /*
     * Create an vector of adjacency lists.
     * Size of vector will be V
     * Initialize each adjacency list as empty by making head as NULL
     */
    this->list_ = std::vector<AdjList>(this->v_, nullptr);
}

Graph::~Graph()
{
    for (auto iter : this->list_)
    {
        auto *ptr = iter.head;
        while (ptr)
        {
            auto prev = ptr;
            ptr = ptr->next;
            delete prev;
        }
    }
}

/*
 * Adds an edge to an undirected graph
 */
void Graph::addEdge(const int src, const int dest, const int weight)
{
    AdjListNode* newNode = nullptr;

    // Add an edge from src to dest.  A new node is added to the adjacency
    // list of src.  The node is added at the begining
    newNode               = new AdjListNode(dest, weight);
    newNode->next         = this->list_[src].head;
    this->list_[src].head = newNode;
 
    // Since graph is undirected, add an edge from dest to src also
    newNode                = new AdjListNode(src, weight);
    newNode->next          = this->list_[dest].head;
    this->list_[dest].head = newNode;
}

/*
 * Method to print shortest path from source to idx using parent vector
 */
void Graph::printPath(const std::vector<int> &parent, int idx)
{
    // Base Case : If j is source
    if (parent[idx]== -1)
    {
        return;
    }
 
    printPath(parent, parent[idx]);

    std::cout << idx << " ";
}

/*
 * A utility method used to print the constructed distance
 */
void Graph::printSolution(const std::vector<int> &dist,
                          const std::vector<int> &parent,
                          const int src)
{
    std::cout << "Source\tDestination\tDistance\tPath" << std::endl;
/*
    for (auto i = dist.size(); i < static_cast<int>(dist.size()); ++i)
    {
        std::cout << src << "\t" << i << "\t\t"
                  << dist[i] << "\t\t" << src << " ";

        if (static_cast<int>(dist.size()) < 100)
        {
            this->printPath(parent, i);
        }
        std::cout << std::endl;
    }
*/
    std::cout << src << "\t" << dist.size() - 1 << "\t\t"
              << dist[dist.size() - 1] << "\t\t" << "n/a"
              << std::endl;
}

/*
 * The main function that calulates distances of shortest paths from src to all
 * vertices. It is a O(ELogV) function
 */
void Graph::dijkstra(const int src)
{
    const auto V = this->v_;  // Get the number of vertices in graph
    std::vector<int> dist(V); // dist values used to pick minimum weight edge in cut

    // Parent array to store shortest path tree
    std::vector<int> parent(V, 0);
    parent[0] = -1;

    /*
     * minHeap represents set E
     * Initially size of min heap is equal to V
     */
    MinHeap minHeap(V);

    /*
     * Initialize min heap with all vertices. dist value of all vertices
     */
    for (auto v = 0; v < V; ++v)
    {
        dist[v]         = std::numeric_limits<int>::max();
        minHeap.heap[v] = new MinHeapNode(v, dist[v]);
        minHeap.pos[v]  = v;
    }

    /*
     * Make dist value of src vertex as 0 so that it is extracted first
     */
    delete minHeap.heap[src];
    minHeap.heap[src]  = new MinHeapNode(src, dist[src]);
    minHeap.pos[src]   = src;
    dist[src] = 0;

    minHeap.decreaseKey(src, dist[src]);

    std::chrono::high_resolution_clock::time_point t1 =  std::chrono::high_resolution_clock::now();

    /*
     * In the followin loop, min heap contains all nodes
     * whose shortest distance is not yet finalized.
     */
    while (!minHeap.isEmpty())
    {
        /*
         * Extract the vertex with minimum distance value
         */
        auto* minHeapNode = minHeap.removeMin();
        auto u = minHeapNode->v; // Store the extracted vertex number
 
        /*
         * Traverse through all adjacent vertices of u (the extracted
         * vertex) and update their distance values
         */
        auto* pCrawl = this->list_[u].head;

        while (pCrawl != nullptr)
        {
            auto v = pCrawl->dest;
 
            /*
             * If shortest distance to v is not finalized yet, and distance to v
             * through u is less than its previously calculated distance
             */
            if (minHeap.isInHeap(v) &&
                dist[u] != std::numeric_limits<int>::max() &&
                pCrawl->weight + dist[u] < dist[v])
            {
                parent[v] = u;
                dist[v]   = dist[u] + pCrawl->weight;
 
                /*
                 * Update distance value in min heap also
                 */
                minHeap.decreaseKey(v, dist[v]);
            }
            pCrawl = pCrawl->next;
        }
    }

    std::chrono::high_resolution_clock::time_point t2 =  std::chrono::high_resolution_clock::now();

    /*
     * Print the calculated shortest distances
     */
    this->printSolution(dist, parent, src);

    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Elapsed time is: " << time_span.count() << " seconds" << std::endl;
}
