#include <algorithm>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>

#define RESET   "\033[0m"
#define BLACK   "\033[30m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"

void simulation(int test)
{
    const auto iterations = 1000000;
    const auto tmp        = iterations * (1.0 / 3.0);
    const auto oneThird   = 1.0 / 3.0;
    const auto twoThird   = 2.0 / 3.0;
    long double chiSquare = 0.0;

    std::vector<int> statistic(3, 0);

    // Obtain a time-based seed
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    // Initialize random generator
    std::default_random_engine generator(seed);
    // Initialize space
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (auto i = 0; i < iterations; ++i)
    {
        auto randomNumber = distribution(generator);

        if ((randomNumber < 1.0) && (randomNumber > twoThird))
            statistic[0] += 1;
        else if ((randomNumber > oneThird) && (randomNumber < twoThird))
            statistic[1] += 1;
        else if ((randomNumber > 0.0) && (randomNumber < oneThird))
            statistic[2] += 1;
    }
 
    std::for_each(statistic.begin(), statistic.end(), [&](int value)
    {
        chiSquare += (pow((value - tmp), 2.0)) / tmp;
    });

    std::cout << "[" << test << "]\tX^2 : " << chiSquare;

    if (chiSquare > 3.6649)
        std::cout << RED << "\tTest fail!" << RESET << std::endl;
    else
        std::cout << GREEN << "\tTest pass!" << RESET << std::endl;
}

int main(int argc, char const *argv[])
{
    for (auto i = 1; i < 11; ++i)
    {
        simulation(i);
    }

    return 0;
}
