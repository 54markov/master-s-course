/*
 * https://en.wikipedia.org/wiki/Ternary_search
 * https://www.hackerearth.com/practice/algorithms/searching/ternary-search/tutorial/
 * https://en.wikipedia.org/wiki/Golden-section_search
 */

#include <chrono>
#include <random>
#include <vector>
#include <limits>
#include <utility>
#include <iostream>
#include <algorithm>

const double goldenSection = 1.61803398875;

double function(double x)
{
    return (std::pow(x, 2) - (5 * x));
}

std::pair<double, double> validation(double left, double right)
{
    double step = 0.001;
    std::vector<std::pair<double, double>> v;

    for (double i = left; i < right; i += step)
    {
        v.push_back(std::make_pair(i, function(i)));
    }

    std::sort(v.begin(), v.end(), [](const std::pair<double, double> &a,
                                     const std::pair<double, double> &b)
    {
        return a.second < b.second; 
    });

    return *(v.begin());
}

void ternarySearch(double (*f)(double),
                   double left,
                   double right,
                   double eps)
{
    auto rv_v = validation(left, right); // return value validation
    double prevValue = 0.0;
    int counter = 0;

    // Search function minimum
    while (right - left > eps)
    {
        double a = right - ((right - left) / goldenSection);
        double b = left  + ((right - left) / goldenSection);

        if (!counter)
        {
            double Fa = f(a);
            double Fb = f(b);

            if (Fa < Fb)
            {
                right     = b;
                prevValue = Fa;
            }
            else
            {
                left      = a;
                prevValue = Fb;
            }
        }
        else
        {
            double Fi = f(b);
            if (Fi < prevValue)
            {
                right     = b;
                prevValue = Fi;
            }
            else
            {
                left      = a;
                prevValue = prevValue;
            }
        }
        counter++;
    }

    double x = (left + right) / 2.0;
    double rv_ts = f(x);

    std::cout << "Fmin (ternary search) : " << rv_ts << ", in point x = ";
    std::cout << x << std::endl;
    std::cout << "Fmin (naive)          : " << rv_v.second << ", in point x = ";
    std::cout << rv_v.first << std::endl;
    std::cout << "*Steps                : " << counter << std::endl;
    std::cout << "*Epsilon              : " << eps << std::endl;
    std::cout << std::endl;
}

int main(int argc, char const *argv[])
{
    double left = -5.0, right = 5.0;

    std::cout << "** Function : x ^ 2 - 5 * x; [-5, 5] **\n" << std::endl;

    ternarySearch(function, left, right, 0.1);
    ternarySearch(function, left, right, 0.001);
    ternarySearch(function, left, right, 0.00001);
    ternarySearch(function, left, right, 0.000001);

    return 0;
}
