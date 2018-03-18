#include <cmath>
#include <tuple>
#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>

#include "color.h"

const auto Z = 2.0 / (1.0 + std::sqrt(5.0));
auto counter = 0;

/*****************************************************************************/
/* Function 1 : x ^ 2 - x * y + y ^ 2 - 2 * x                                */
/*****************************************************************************/
template <typename T>
T f1(T x, T y)
{
    return (std::pow(x, 2.0) - x * y + std::pow(y, 2.0) - 2.0 * x);
}

template <typename T>
std::vector<T> gradF1(T x, T y)
{
    return std::vector<T> { 2.0 * x - y - 2.0, -x + 2.0 * y };
}

/*****************************************************************************/
/* Function 2 : x ^ 2 - x * y + y ^ 2 - 2 * x                                */
/*****************************************************************************/
template <typename T>
T f2(T x, T y)
{
    return (std::pow(x - y * y, 2.0) + 0.1 * std::pow(x - 0.7, 2.0));
}

template <typename T>
std::vector<T> gradF2(T x, T y)
{
    return std::vector<T> { 2.2 * x - 2.0 * std::pow(y, 2.0) - 0.14, 
                            4.0 * x * y + 4.0 * std::pow(y, 3.0) };
}

/*****************************************************************************/
/* Function 3 : (x - 3) ^ 2 + (y - 4) ^ 2                                    */
/*****************************************************************************/
template <typename T>
T f3(T x, T y)
{
    return std::pow(x - 3.0, 2.0) + std::pow(y - 4.0, 2.0);
}

template <typename T>
std::vector<T> gradF3(T x, T y)
{
    return std::vector<T> { 2.0 * x - 6.0, 2.0 * y - 8.0 };
}

/*****************************************************************************/
/* Function 4 : (x - 3) ^ 2 + 2 * (y - 4) ^ 2                                */
/*****************************************************************************/
template <typename T>
T f4(T x, T y)
{
    return (std::pow(x - 3.0, 2.0) + 2.0 * std::pow(y - 4.0, 2.0));
}

template <typename T>
std::vector<T> gradF4(T x, T y)
{
    return std::vector<T> { 2.0 * x - 6.0, 4.0 * y - 16.0 };
}

/*****************************************************************************/
/* Naive validation                                                          */
/*****************************************************************************/
class Container
{
    public:
        double x, y, f;
        Container(double _x, double _y, double _f = 0.0) : x(_x), y(_y), f(_f) {}
};

template <typename T>
void validation(T (* f)(T, T), std::pair<T, T> x, std::pair<T, T> y)
{
    const auto step = 0.01;

    std::vector<Container> v;

    for (auto i = x.first; i < x.second; i += step)
    {
        for (auto j = y.first; j < y.second; j += step)
        {
            v.push_back(Container(j, i, f(j, i)));
        }
    }

    std::sort(v.begin(), v.end(), [](const auto &a, const auto &b)
    {
        return a.f < b.f;
    });

    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    std::cout << green;
    std::cout << "*** Validation (naive) ***" << std::endl;
    std::cout << "x       : " << (v.begin())->x << std::endl;
    std::cout << "y       : " << (v.begin())->y << std::endl;
    std::cout << "f(x, y) : " << (v.begin())->f << std::endl;
    std::cout << def;

    return;
}

template <typename T>
T findLamb(T (* function)(T, T), T x, T y, std::vector<T> grad, T eps)
{
    auto e     = 1E-10;
    auto a     = 0.0;
    auto b     = 1.0;
    auto lamb1 = b - (b - a) * Z;
    auto lamb2 = a + (b - a) * Z;
    auto y1    = function(x - lamb1 * grad[0], y - lamb1 * grad[1]);
    auto y2    = function(x - lamb2 * grad[0], y - lamb2 * grad[1]);

    while (true)
    {
        if (y1 >= y2)
        {
            a     = lamb1;
            lamb1 = lamb2;
            y1    = y2;
            lamb2 = a + (b - a) * Z;
            y2    = function(x - lamb2 * grad[0], y - lamb2 * grad[1]);
        }
        else
        {
            b     = lamb2;
            lamb2 = lamb1;
            y2    = y1;
            lamb1 = b - (b - a) * Z;
            y1    = function(x - lamb1 * grad[0], y - lamb1 * grad[1]);
        }

        if (std::abs(b - a) < e)
        {
            return (a + b) / 2.0;
        }
    }
}

template <typename T>
std::tuple<int, T, T> gradientMethod(T (* function)(T, T),
                                     std::vector<T> (* gradient)(T, T),
                                     T x,
                                     T y,
                                     const T eps)
{
    Color::Modifier red(Color::FG_RED);
    Color::Modifier def(Color::FG_DEFAULT);

    auto iteration = 0;
    auto result    = 0.0;

    auto subXprev = 0.0;
    auto subYprev = 0.0;

    do
    {
        iteration++;

        auto x1   = x;
        auto y1   = y;
        auto grad = gradient(x, y);
        auto lamb = findLamb(function, x, y, grad, eps);

        x = x - lamb * grad[0];
        y = y - lamb * grad[1];

        result = std::sqrt(std::pow(x - x1, 2.0) + std::pow(y - y1, 2.0));

        auto mulX = subXprev * (x - x1);
        auto mulY = subYprev * (y - y1);

        subXprev = x - x1; // Rember value
        subYprev = y - y1; // Rember value

        std::cout << red;
        std::cout << "dx         : " << subXprev << std::endl;
        std::cout << "dy         : " << subYprev << std::endl;
        std::cout << "Scalar add : " << mulX + mulY << std::endl;
        std::cout << std::endl;
        std::cout << def;


    } while (result >= eps && iteration < 999999);

    return std::make_tuple(iteration, x, y);
}

template <typename T>
void test(T (* f)(T, T), std::vector<T> (* g)(T, T), const T x, const T y, const T eps)
{
    try
    {
        counter = 0;
        auto rv = gradientMethod(f, g, x, y, eps);

        std::cout << "Start point (" << x << ", " << y << ")" << std::endl;

        std::cout << "Epsilon : " << eps << std::endl
                  << "Counter : " << std::get<0>(rv) << std::endl
                  << "x       : " << std::get<1>(rv) << std::endl
                  << "y       : " << std::get<2>(rv) << std::endl
                  << "f(x, y) : " << f(std::get<1>(rv), std::get<2>(rv)) << std::endl
                  << std::endl;
    }
    catch (std::string err)
    {
        std::cerr << err << std::endl;
    }
}

int main(int argc, char const *argv[])
{
    Color::Modifier yellow(Color::FG_YELLOW);
    Color::Modifier def(Color::FG_DEFAULT);

    std::cout << yellow;
    std::cout << "*** Running function 4 : (x - 3) ^ 2 + 2 * (y - 4) ^ 2 ***" << std::endl;
    std::cout << def;
    test(f4, gradF4, 8.0, 8.0, 1E-1);
    test(f4, gradF4, 8.0, 8.0, 1E-3);
    test(f4, gradF4, 8.0, 8.0, 1E-6);

    test(f4, gradF4, 3.0, 8.0, 1E-1);
    test(f4, gradF4, 3.0, 8.0, 1E-3);
    test(f4, gradF4, 3.0, 8.0, 1E-6);

    validation(f4, std::make_pair(0.0, 4.0), std::make_pair(0.0, 4.0));

    std::cout << std::endl;

    std::cout << yellow;
    std::cout << "*** Running function 1 : x ^ 2 - x * y + y ^ 2 - 2 * x ***" << std::endl;
    std::cout << def;
    test(f1, gradF1, 8.0, 8.0, 1E-1);
    test(f1, gradF1, 8.0, 8.0, 1E-3);
    test(f1, gradF1, 8.0, 8.0, 1E-6);

    test(f1, gradF1, 4.0, 4.0, 1E-1);
    test(f1, gradF1, 4.0, 4.0, 1E-3);
    test(f1, gradF1, 4.0, 4.0, 1E-6);

    validation(f1, std::make_pair(0.0, 4.0), std::make_pair(0.0, 4.0));

    return 0;
}