#ifndef _MFS_H_
#define _MFS_H_

#include "graph.h"

/*
 * MSF - minimum spanning tree
 */

typedef struct
{
    int size;
    int first;
} set_t;

typedef struct
{
    int set;
    int next;
} elem_t;

class MFSet
{
    public:
        MFSet(const int nelems);
        ~MFSet();

        void makeSet(int elem);
        void unionSet(int elem1, int elem2);
        int findSet(const int elem);
        int searchKruskal(Graph *g, Graph *mst);

    private:
        int     nelems_;
        int     nsets_;
        set_t  *sets_;
        elem_t *elems_;
};

#endif /* _MFS_H_ */
