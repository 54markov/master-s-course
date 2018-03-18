#ifndef _GRAPH_H_
#define _GRAPH_H_

/*
 * Реализация графа на базе матрицы смежности
 */
class Graph
{
    public:
        Graph(const int nvertices);
        ~Graph();

        int getEdge(const int i, const int j);
        void setEdge(const int i, const int j, int w);
        void clear();
        int getVerctices();

    private:
        int  nvertices_; /* Число вершин */
        int *m_;         /* Матрица n x n */
        int *visited_;
};

#endif /* _GRAPH_H_ */
