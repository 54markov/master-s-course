#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "color.h"
#include "calendar.h"
#include "statistic.h"

void test(int testNum, int modelingTime, std::vector<double> lambda)
{
    Statistic statistic(testNum, lambda[0], lambda[1], lambda[2], lambda[3]);

    double startTime = 0.0;

    resetModelingTime();
    resetQueues();

    // Planing events before start
    if (testNum == 1)
    {
        schedule(E1, startTime + getRandExp(lambda[0]));
        schedule(E2, startTime + getRandExp(lambda[1]));
        schedule(E3, startTime + getRandExp(lambda[2]));
        schedule(E4, startTime + getRandExp(lambda[3]));
    }
    else if (testNum == 2)
    {
        schedule(E1, startTime + getRandExp(lambda[0]));
        schedule(E2, startTime + getRandExp(lambda[1]));
        schedule(E3, startTime + (1.0 / lambda[2]));
        schedule(E4, startTime + (1.0 / lambda[3]));
    }

    // Run actual simlation
    simulate(statistic, testNum, modelingTime, lambda);

    statistic.showStatistic(); 

    return;
}

int main(int argc, char const *argv[])
{
    try
    {
        Color::Modifier yellow(Color::FG_YELLOW);
        Color::Modifier def(Color::FG_DEFAULT);
        if (argc != 2)
        {
            throw std::string("usage " + std::string(argv[0]) + " <file-name>");
        }

        std::ifstream file(argv[1]);
        if (!file.is_open())
        {
            throw std::string("Can't open file");
        }

        /*
         * Read only four lines form file
         * Floating point format required
         * - line 1 : lambda 1
         * - line 2 : lambda 2
         * - line 3 : lambda 3
         * - line 4 : lambda 4
         */
        std::vector<double> lambdas;

        for (auto i = 0; i < 4; ++i)
        {
            std::string::size_type sz;
            std::string line;

            getline(file, line);
            std::cout << yellow << "Reading from file : " << line << def << std::endl;
            
            lambdas.push_back(std::stod(line, &sz));
        }

        test(1, 1000.0, lambdas); // Running test 1
        test(2, 1000.0, lambdas); // Running test 2

    }
    catch (std::string err)
    {
        Color::Modifier red(Color::FG_RED);
        Color::Modifier def(Color::FG_DEFAULT);
        std::cerr << red << err << def << std::endl;
        return -1;
    }
    return 0;
}