/*
 * http://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/
 * C++ program for implementation of Ford Fulkerson algorithm
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <limits>
#include <string>
#include <chrono>
#include <queue>

void printGraph(std::vector<std::vector<int>> &graph)
{
    char vertex = 'A';

    for (auto i = 0; i < static_cast<int>(graph.size()); ++i)
    {
        char a = vertex + i;
        std::cout << a << "-+" << std::endl;
        for (auto j = 0; j < static_cast<int>(graph[i].size()); ++j)
        {
            if (graph[i][j] > 0)
            {
                char b = vertex + j;
                std::cout << "  +->"<< b << " : " << graph[i][j] << std::endl;
            }
        }
        std::cout << std::endl;
    }
}

/*
 * Возвращает true, если есть путь от источника 's' в сток 't' в
 * остаточный граф.
 * Также заполням вектор parent[], чтобы сохранить путь
 */
bool bfs(std::vector<std::vector<int>> rGraph, int s, int t, std::vector<int> &parent)
{
    /*
     * Поиск в ширину (breadth-first search – BFS, обход в ширину)
     * – процедура посещения всех вершин графа
     * начиная с заданного узла "v"
     * Сперва посещаем свои дочерние вершины
     */

    const int V = rGraph.size();

    /*
     * Создайте посещенный массив и отмечаем все вершины как не посещенные
     */
    std::vector<bool> visited(V, false);

    /*
     * Создаем очередь, помещаем вершину источника и отмечаем как посещенную
     */
    std::queue <int> q;
    q.push(s);

    visited[s] = true;
    parent[s]  = -1;

    while (!q.empty())
    {
        auto u = q.front();
        q.pop();

        for (auto v = 0; v < V; v++)
        {
            if (visited[v] == false && rGraph[u][v] > 0)
            {
                q.push(v);
                parent[v]  = u;
                visited[v] = true;
            }
        }
    }

    /*
     * Если мы достигли стока, true иначе false
     */
    return (visited[t] == true);
}

/*
 * Возвращает максимальный поток от 's' до 't' в данном графе
 */
int fordFulkerson(std::vector<std::vector<int>> graph, int s, int t)
{
    auto iter = 0;
    auto maxFlow = 0;
    const auto V = graph.size();
    /*
     * Создаем остаточный граф и заполняем его с помощью
     * заданной мощности в исходном графе
     */
    std::vector<std::vector<int>> rGraph = graph;
    /*
     * Остаточный граф, где rGraph [i] [j] показывает
     * остаточную емкость ребра от i до j (если там - ребро).
     * Если rGraph [i] [j] равно 0, (то нет ребра)
     */

    /*
     * Этот вектор заполняется BFS и сохраняет путь
     */
    std::vector<int> parent(V, 0);

    //printGraph(rGraph);

    std::chrono::high_resolution_clock::time_point t1 =  std::chrono::high_resolution_clock::now();
    /*
     * Увеличиваем поток, пока есть путь от источника к потоку
     */
    while (bfs(rGraph, s, t, parent))
    {
        /*
         * Находим минимальную остаточную емкость ребер вдоль пути,
         * заполненного BFS. 
         * Или же мы можем сказать, что нашли максимальный поток
         * в найденном пути.
         */
        auto path_flow = std::numeric_limits<int>::max();
        for (auto v = t; v != s; v = parent[v])
        {
            auto u    = parent[v];
            path_flow = std::min(path_flow, rGraph[u][v]);
        }

        /*
         * Обновяем остаточные ёмкости ребер и
         * обратные ребра вдоль пути
         */

        std::vector<int> path;
        for (auto v = t; v != s; v = parent[v])
        {
            auto u        = parent[v];
            rGraph[u][v] -= path_flow;
            rGraph[v][u] += path_flow;

            path.push_back(u);
            //std::cout << u <<  " ";
        }

        char last = (V - 1) + 'A';
        std::cout << last << " ";
        for (auto i : path)
        {
            std::cout << (char)('A' + i) <<  " ";   
        }

        std::cout << std::endl;

        /*
         * Добавляем поток пути к общему потоку
         */
        maxFlow += path_flow;

        /*
         * Печатаем полученные значения
         */
        std::cout << "Iteration " << iter++
                  << " "
                  << "max flow " << path_flow
                  << std::endl;

        //printGraph(rGraph);
    }

    std::chrono::high_resolution_clock::time_point t2 =  std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Elapsed time is: " << time_span.count() << " seconds" << std::endl;
    /*
     * Возвращаем общий поток
     */
    return maxFlow;
}

void testGraph1()
{
    const int source      = 0;
    const int destination = 5;

    std::vector<std::vector<int>> graph;

    graph.push_back({ 0, 16, 13, 0, 0, 0 });
    graph.push_back({ 0, 0, 10, 12, 0, 0 });
    graph.push_back({ 0, 4, 0, 0, 14, 0 });
    graph.push_back({ 0, 0, 9, 0, 0, 20 });
    graph.push_back({ 0, 0, 0, 7, 0, 4 });
    graph.push_back({ 0, 0, 0, 0, 0, 0 });

    std::cout << "The maximum possible flow is "
              << fordFulkerson(graph, source, destination)
              << std::endl;    
}

void testGraph2()
{
    const int source      = 0;
    const int destination = 7;

    std::vector<std::vector<int>> graph;

    //                 0   1  2  3   4  5   6  7
    graph.push_back( { 0, 10, 5, 15, 0, 0,  0,  0  } ); // 0
    graph.push_back( { 0, 0,  4, 0,  9, 15, 0,  0  } ); // 1
    graph.push_back( { 0, 0,  0, 4,  0, 8,  0,  0  } ); // 2
    graph.push_back( { 0, 0,  0, 0,  0, 0,  16, 0  } ); // 3
    graph.push_back( { 0, 0,  0, 0,  0, 15, 0,  10 } ); // 4
    graph.push_back( { 0, 0,  0, 0,  0, 0,  15, 10 } ); // 5
    graph.push_back( { 0, 0,  6, 0,  0, 0,  0,  10 } ); // 6
    graph.push_back( { 0, 0,  0, 0,  0, 0,  0,  0  } ); // 7

    std::cout << "The maximum possible flow is "
              << fordFulkerson(graph, source, destination)
              << std::endl;
}

void testGraph3()
{
    int source      = 0;
    int destination = 0;

    std::vector<std::vector<int>> graph;

    std::ifstream file("build/graph.txt");

    try
    {
        if (!file.is_open())
        {
            throw std::string("Can't read from file");
        }

        std::string line;
        getline(file, line);

        std::cout << "Graph size is : " << line << std::endl;

        auto vertecies = (std::stoi(line));

        for (auto i = 0; i < vertecies; ++i)
        {
            graph.push_back(std::vector<int>(vertecies, 0));
        }

        while (getline(file, line))
        {
            int src, dst, weight;
            std::stringstream ss;
            ss << line;
            ss >> src;
            ss >> dst;
            ss >> weight;

            //std::cout << "Added edge: "
            //          << src << " "
            //          << dst << " "
            //          << weight << std::endl;

            graph[src][dst] = weight; // Set edge
        }

        file.close();

        for (auto i: graph)
        {
            for (auto j: i)
            {
                std::cout << j << "\t";
            }
            std::cout << std::endl;
        }

        std::cout << "Running Ford-Fulkerson algorithm for Maximum-Flow-Problem:" << std::endl;
        std::cout << "Graph:" << std::endl;
        std::cout << " (0)---1---(1)---1---(n)  " << std::endl;
        std::cout << " |          |         |  " << std::endl;
        std::cout << " |          |         |  " << std::endl;
        std::cout << " |          |         |  " << std::endl;
        std::cout << " 1          1         1   " << std::endl;
        std::cout << " |          |         |  " << std::endl;
        std::cout << " |          |         |  " << std::endl;
        std::cout << " |          |         |  " << std::endl;
        std::cout << " (.)---1---(.)---1---(n)  " << std::endl;
        std::cout << "Source vertex is '" <<  source << "'" << std::endl;

        destination = vertecies - 1;

        std::cout << "The maximum possible flow is "
                  << fordFulkerson(graph, source, destination)
                  << std::endl;
    }
    catch (std::string err)
    {
        std::cerr << err << std::endl;
        return;
    }
}

int main(int argc, char const *argv[])
{
    //testGraph1();
    
    //testGraph2();

    testGraph3();
 
    return 0;
}
