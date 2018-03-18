#include <iostream>
#include <fstream>
#include <string>
#include <vector>

typedef struct
{
    int src;
    int dst;
    int weight;
} adjacencyList_t;

void generateGraph(const int size)
{
    auto vertex = 0;

    std::ofstream file;
    
    std::vector<adjacencyList_t> v;

    for (auto i = 0; i < size - 1; ++i)
    {
        for (auto j = 0; j < size - 1; ++j)
        {
            auto right = vertex + 1;
            auto down  = vertex + size;
            v.push_back( {.src = vertex, .dst = right, .weight = 1} );
            v.push_back( {.src = vertex, .dst = down,  .weight = 1} );
            vertex++;
        }

        // Generate last graph col
        auto down = vertex + size;
        v.push_back( {.src = vertex, .dst = down, .weight = 1} );
        vertex++;
    }

    // Generate last graph row
    for (auto j = 0; j < size - 1; ++j)
    {
        auto right = vertex + 1;
        v.push_back( {.src = vertex, .dst = right, .weight = 1} );
        vertex++;
    }

    file.open("build/graph.txt");
    try
    {
        if (!file.is_open())
            throw std::string("Can't create file");

        file << size * size << std::endl;

        for (auto i : v)
        {
/*
            std::cout << "Source: " << i.src << "\t"
                      << "Destination: " << i.dst << "\t"
                      << "Weight:" << i.weight << std::endl;
*/
            file << i.src << " " << i.dst << " " << i.weight << std::endl;
        }
        file.close();
    }
    catch (std::string err)
    {
        std::cerr << err << std::endl;
        return;
    }
}

int main(int argc, char const *argv[])
{
    const auto size = 10;

    generateGraph(size);

    return 0;
}
