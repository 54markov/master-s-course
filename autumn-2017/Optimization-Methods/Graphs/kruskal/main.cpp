#include "mfs.h"

#include <string>
#include <iostream>

int main(int argc, char const *argv[])
{
    try
    {
        Graph graph(6);
        Graph mst(6);
        MFSet mfset(6);

        graph.setEdge(1, 2, 6);
        graph.setEdge(1, 3, 1);
        graph.setEdge(1, 4, 5);
        graph.setEdge(2, 3, 5);
        graph.setEdge(2, 5, 3);
        graph.setEdge(3, 4, 5);
        graph.setEdge(3, 5, 6);
        graph.setEdge(3, 6, 4);
        graph.setEdge(4, 6, 2);
        graph.setEdge(5, 6, 6);

        std::cout << "Runnig the Kruskal algorithm" << std::endl;
        std::cout << "Graph:" << std::endl;
        std::cout << " +-------A-------+ " << std::endl;
        std::cout << " |       |       | " << std::endl;
        std::cout << " 6       1       5 " << std::endl;
        std::cout << " |       |       | " << std::endl;
        std::cout << " B---5---C---5---D " << std::endl;
        std::cout << " |      / \\      | " << std::endl;
        std::cout << " 3     6   4     2 " << std::endl;
        std::cout << " |    /     \\    | " << std::endl;
        std::cout << " +---E---6---F---+ " << std::endl << std::endl;
         
        mfset.searchKruskal(&graph, &mst);
    }
    catch (std::string err)
    {
        std::cerr << err << std::endl;
        return -1;
    }

    return 0;
}
