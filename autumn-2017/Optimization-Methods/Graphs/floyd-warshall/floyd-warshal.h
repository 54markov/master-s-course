#ifndef _FLOYD_WARSHALL_H_
#define _FLOYD_WARSHALL_H_

#include <vector>

/*
 * Define Infinite as a large enough value.
 * This value will be used for vertices not connected to each other
 */
const auto INF = 99999;

class FloydWarshal
{
    public:
        FloydWarshal();
        ~FloydWarshal();

        void createTestGraph(int vertices);
        void printSolution(std::vector<std::vector<int>> &v);
        void floydWarshall();

    private:
        int vertices_;
        std::vector<std::vector<int>> graph_;
};

#endif /* _FLOYD_WARSHALL_H_ */
