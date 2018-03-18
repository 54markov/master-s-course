#ifndef _MIN_HEAP_H_
#define _MIN_HEAP_H_

/*
 * Class to represent a min-heap-node
 */
class MinHeapNode
{
    public:
        MinHeapNode(const int _v, const int _dist);
        ~MinHeapNode();
        
        int v;
        int dist;
};
 
/*
 * Class to represent a min-heap
 */
class MinHeap
{
    public:
        MinHeap(const int capacity);
        ~MinHeap();

        /*
         * A standard method to heapify at given idx
         * This method also updates position of nodes when they are swapped.
         * Position is needed for decreaseKey()
         */
        void heapify(const int idx);
        /*
         * Standard method to extract minimum node from heap
         */
        MinHeapNode *removeMin();
        /*
         * Method to decreasy dist value of a given vertex v.
         * This Method uses pos[] of min heap to get the current index
         * of node in min heap
         */
        void decreaseKey(int v, int dist);

        /*
         * A utility method to swap two nodes of min heap (needed for min heapify)
         */
        //void swap(MinHeapNode** a, MinHeapNode** b);
        void swap(MinHeapNode &a, MinHeapNode &b);
        /*
         * A utility method to check if the given minHeap is empty or not
         */
        int isEmpty();
        /*
         * A utility method to check if a given vertex 'v' is in min heap or not
         */
        bool isInHeap(const int v);

        int                  counter;  // Number of heap nodes present currently
        int                  capacity; // Capacity of min heap
        int                 *pos;      // This is needed for decreaseKey()
        struct MinHeapNode **heap;
};
 
#endif /* _MIN_HEAP_H_ */
