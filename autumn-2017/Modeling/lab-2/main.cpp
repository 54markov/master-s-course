#include <thread>
#include <vector>
#include <string>
#include <iostream>

#include "task.h"
#include "worker.h"
#include "manager.h"
#include "thread-safe-queue.h"
#include "thread-safe-container.h"

const int GLOBAL_CYCLE = 100;

/*****************************************************************************/
/*                                                                           */      
/*      +-------------+                                                      */
/*     \|/            |                                                      */
/* --->[Q]---------->[W1]---------->[W2]---------->                          */
/*          w1=work  /|\   w1=free   |    w2=free                            */
/*                    |              |                                       */
/*                    +--------------+                                       */
/*                         w2=free                                           */
/*****************************************************************************/

void threadWorker(int                                   num,
                  double                                param,
                  TSQueue                              *q,
                  TSContainer                          *c,
                  std::vector <std::pair<int, double>> *s1,
                  std::vector <std::pair<int, double>> *s2)
{
    Worker worker(GLOBAL_CYCLE, num, param, q, c, s1, s2);
    worker.doWork();
}

void runTest(int testNumber)
{
    std::vector <std::pair<int, double>> statistic0;
    std::vector <std::pair<int, double>> statistic1;
    std::vector <std::pair<int, double>> statistic2;

    TSQueue queue;
    TSContainer container;

    double lambda = 2.0;

    // Task queue manager
    Manager manager(lambda, GLOBAL_CYCLE, &statistic0);
    
    double intensity1 = 0.0;
    double intensity2 = 0.0;

    if (testNumber == 0)
    {
        intensity1 = 1.5;
        intensity2 = 3.0;
        std::cout << "Run test worker1 = 1.5, worker2 = 3.0 " << std::endl;
    }
    else
    {
        intensity1 = 3.0;
        intensity2 = 1.5;
        std::cout << "Run test worker1 = 3.0, worker2 = 1.5 " << std::endl;   
    }

    std::thread thWorker1(threadWorker, 1, intensity1,
        &queue, &container, &statistic1, &statistic2);
    
    std::thread thWorker2(threadWorker, 2, intensity2,
        &queue, &container, &statistic1, &statistic2);

    manager.doWork(queue);

    thWorker1.join();
    thWorker2.join();

#ifdef DEBUG
    std::cout << std::endl;
    std::cout << statistic0.size() << std::endl;
    std::cout << statistic1.size() << std::endl;
    std::cout << statistic2.size() << std::endl;
    std::cout << std::endl;

    std::cout << "Queue\t    Worker1\t    Worker2\t    ElapsedTime" << std::endl;

    for (auto i = 0; i < GLOBAL_CYCLE; ++i)
    {
            std::cout << statistic0[i].second << "\t    ";
            std::cout << statistic1[i].second << "\t    ";
            std::cout << statistic2[i].second << "\t    ";
            std::cout << statistic2[i].second - statistic0[i].second << std::endl;
    }
#endif /* DEBUG */

    std::vector<double> v;
    double meanTime = 0.0;

    for (auto i = 0; i < GLOBAL_CYCLE; ++i)
    {
        v.push_back(statistic2[i].second - statistic0[i].second);
    }

    std::for_each(v.begin(), v.end(), [&](double value)
    {
        meanTime += value;
    });

    meanTime /= GLOBAL_CYCLE;

    std::cout << "Mean time   : " << meanTime << std::endl;
    std::cout << "Litle's law : " << lambda * meanTime << std::endl;

    if (!queue.is_empty())
        throw std::string("Global queue - non empty!");
}

int main(int argc, char const *argv[])
{
    try
    {
        if (argc < 2)
            throw std::string("Not enought arguments");

        std::string option(argv[1]);

        if (!option.compare("a"))
            runTest(0);
        else if (!option.compare("b"))
            runTest(1);
        else
            throw std::string("Not recognized option");
    }
    catch (std::string err)
    {
        std::cerr << err << std::endl;
        return -1;
    }

    return 0;
}