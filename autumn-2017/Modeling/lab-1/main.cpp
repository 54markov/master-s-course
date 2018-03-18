#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>

#include <math.h>

using namespace std;

const auto iterations = 1000000;

void fillVectors(void)
{
    // Obtain a time-based seed
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();   
    // Initialize random generator
    std::default_random_engine generator(seed);
    // Initialize random generator space
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    std::vector<double> interval(10);
    std::vector<int>    statistic(10);
    interval  = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    statistic = { 0,   0,   0,   0,   0,   0,   0,   0,   0,   0};

    for (auto i = 0; i < iterations; ++i)
    {
        auto randomNumber = distribution(generator);

        if ((0.0 < randomNumber) && (randomNumber < interval[0]))
        {
            statistic[0] += 1;
        }

        for (auto j = 1; j < (int)interval.size(); j++)
        {
            if ((interval[j - 1] < randomNumber) && (randomNumber < interval[j]))
            {
                statistic[j] += 1;
            }
        }
    }

    long double sum = 0.0;
/*
    std::for_each(statistic.begin(), statistic.end(), [](int& value) {
        std::cout << value << std::endl;
    });
*/
    for (auto i = 0; i < (int)statistic.size(); ++i)
    {
        double tmp = iterations * 0.1;
        sum += (pow((statistic[i] - tmp), 2.0)) / tmp;
    }
    std::cout << "Sum : " << sum << std::endl;
}

int main(int argc, char const *argv[])
{
    fillVectors();
    return 0;
}
