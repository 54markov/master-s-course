#ifndef _HEAP_H_
#define _HEAP_H_

struct heapvalue
{
    int i;
    int j;
};

struct heapitem
{
    int priority;
    struct heapvalue value;
};

class Heap
{
    public:
        Heap(int maxsize);
        ~Heap();

        void swap(struct heapitem *a, struct heapitem *b);
        int insert(int priority, struct heapvalue value);
        struct heapitem removeMin();
        void heapify(int index);

    private:
        int maxsize_;            /* Array size */
        int nnodes_;             /* Number of keys */
        struct heapitem *nodes_; /* Nodes: [0..maxsize] */
};

#endif /* _HEAP_H_ */