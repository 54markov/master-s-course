#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>

const double goldenSection = 1.61803398875;

class Container
{
    public:
        Container(double _x, double _y, double _f) : x(_x), y(_y), f(_f) { }
        double x, y, f;
};

double function1(double x, double y)
{
    return (std::pow(x, 2) - (x * y) + std::pow(y, 2) - (2 * x));
}

double function2(double x, double y)
{
    return (std::pow(x - std::pow(y, 2), 2) + 0.1 * std::pow((x - 0.7), 2));
}

Container validation(auto (* f)(auto, auto), auto &x, auto &y)
{
    auto step = 0.01;
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

    return *(v.begin());
}

double nestedTernarySearch_(auto (* f)(auto, auto), auto i,
                            auto &y, auto eps, auto &counter, auto &pointY)
{
    auto left      = y.first;
    auto right     = y.second;
    auto prevValue = 0.0;

    while (std::abs(right - left) > eps)
    {
        auto a = right - ((right - left) / goldenSection);
        auto b = left  + ((right - left) / goldenSection);
        auto firstInvoke = true;

        if (firstInvoke)
        {
            auto Fa = f(i, a);
            auto Fb = f(i, b);

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
            firstInvoke = false;
        }
        else
        {
            auto Fi = f(i, b);
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

    pointY = ((left + right) / 2.0);

    return f(i, ((left + right) / 2.0));
}

void nestedTernarySearch(auto (* f)(auto, auto), auto &x, auto &y, auto eps)
{
    auto rv_v = validation(f, x, y);
    auto counter = 0;

    std::vector<Container> v;

    auto left  = x.first;
    auto right = x.second;

    // Search function minimum
    while (std::abs(right - left) > eps)
    {
        auto a = right - ((right - left) / goldenSection);
        auto b = left  + ((right - left) / goldenSection);
        auto pointY = 0.0;

        auto Fa = nestedTernarySearch_(f, a, y, eps, counter, pointY);
        auto Fb = nestedTernarySearch_(f, b, y, eps, counter, pointY);

        if (Fa < Fb)
        {
            right = b;
        }
        else
        {
            left = a;
        }

        auto pointX = ((left + right) / 2.0);

        v.push_back(Container(pointX, pointY, f(pointX, pointY)));
    }

    std::sort(v.begin(), v.end(), [](const auto &a, const auto &b)
    {
        return a.f < b.f; 
    });

    auto rv_ts = *v.begin();

    std::cout << "Fmin (nested teranry search) : " << rv_ts.f;
    std::cout << "\tin point (" << rv_ts.x << ", " <<  rv_ts.y << ")" << std::endl;
    std::cout << "Fmin (naive)                 : " << rv_v.f;
    std::cout << "\tin point (" << rv_v.x << "," <<  rv_v.y << ")" << std::endl;
    std::cout << "*Steps                       : " << counter << std::endl;
    std::cout << "*Epsilon                     : " << eps << std::endl;
    std::cout << std::endl;
}

int main(int argc, char const *argv[])
{
    std::pair<double,double> x1(-5.0, 5.0);
    std::pair<double,double> y1(-5.0, 5.0);
    std::pair<double,double> x2(-5.0, 5.0);
    std::pair<double,double> y2(-5.0, 5.0);

    std::cout << "\n ** Function : x ^ 2 - x * y + y ^ 2 - 2 * x; [-5, 5], [-5, 5] **\n" << std::endl;
    nestedTernarySearch(function1, x1, y1, 0.01);
    nestedTernarySearch(function1, x1, y1, 0.0001);
    nestedTernarySearch(function1, x1, y1, 0.000001);

    std::cout << "\n ** Function : (x - y ^ 2) ^ 2 + 0.1 * (x - 0.7) ^ 2; [-5, 5], [-5, 5] **\n" << std::endl;
    nestedTernarySearch(function2, x2, y2, 0.01);
    nestedTernarySearch(function2, x2, y2, 0.0001);
    nestedTernarySearch(function2, x2, y2, 0.000001);

    return 0;
}
