#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>

#include <stdio.h>

const double GOLDEN_SECTION = 1.61803398875;
double EPS = 1e-3;

/*****************************************************************************/
/* Function: (x - 3) ^ 2 + 2 * (y - 4) ^ 2                                   */
/*****************************************************************************/
template <typename T>
T function(T x, T y)
{
    return (std::pow(x - 3.0, 2.0) + 2.0 * std::pow(y - 4.0, 2.0));
}

/*****************************************************************************/
/* Container                                                                 */
/*****************************************************************************/
class Container
{
    public:
        double x, y, f;

        Container(double _x, double _y, double _f = 0.0) : x(_x), y(_y), f(_f) {}

        Container operator = (Container rhs);
        Container getDerivative();
};

Container Container::operator = (Container rhs)
{
    this->x = rhs.x,
    this->y = rhs.y,
    this->f = rhs.f;

    return *this;
}

Container Container::getDerivative()
{
    return Container((2.0 * this->x - 6.0), (2.0 * this->y - 8.0));
}

/*****************************************************************************/
/* Naive validation                                                          */
/*****************************************************************************/
template <typename T>
Container validation(T (* f)(T, T), std::pair<T, T> x, std::pair<T, T> y)
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

    std::cout << "Validation (naive)" << std::endl;
    std::cout << "F(min) : " << (v.begin())->f << std::endl;
    std::cout << "X(min) : " << (v.begin())->x << std::endl; 
    std::cout << "Y(min) : " << (v.begin())->y << std::endl; 

    return *(v.begin());
}

template <typename T>
T getModuleGradient(T x, T y)
{
    return std::sqrt(std::pow(x, 2.0) + std::pow(y, 2.0));
}

template <typename T>
T computeNextPoint(T pointValue, T step, T gradientValue)
{
    return (pointValue - (step * gradientValue));
}

template <typename T>
T getGoldenSectionMin(T left, T right, T currentPoint, T gradient, int &counter)
{  
    double x1 = right - ((right - left) / GOLDEN_SECTION);
    double x2 = left  + ((right - left) / GOLDEN_SECTION);

    double gradX1 = currentPoint - x1 * gradient;
    double gradX2 = currentPoint - x2 * gradient;

    double fun1   = 2.0 * gradX1 - 6.0;
    double fun2   = 4.0 * gradX2 - 16.0;

    while (std::abs(fun1 - fun2) > EPS)
    {
        if (std::abs(fun1) > std::abs(fun2))
        {
            left   = x1;
            x1     = x2;
            fun1   = fun2;
            gradX1 = gradX2;
            x2     = left + ((right - left) / GOLDEN_SECTION);
            gradX2 = currentPoint - x2 * gradient;
            fun2   = 4.0 * gradX2 - 16.0;
        }
        else
        {
            right  = x2;
            x2     = x1;
            fun2   = fun1;
            gradX2 = gradX1;
            x1     = right - ((right - left) / GOLDEN_SECTION);
            gradX1 = currentPoint - x1 * gradient;
            fun1   = 2.0 * gradX1 - 6.0;
        }
        counter++;
    }

    return ((x1 + x2) / (2.0));
}

void test(double x, double y)
{
    double stepSize       = 0.0;
    double moduleGradient = 1.0;
    int counter           = 0;

    Container startPoint(x, y);
    Container currentPoint(startPoint.x, startPoint.y);
    auto currentGPoint  = currentPoint.getDerivative();
    
    while (std::abs(moduleGradient) > 0.01)
    {
        // Current Gradient Point
        currentGPoint  = currentPoint.getDerivative();

        moduleGradient = getModuleGradient(currentGPoint.x, currentGPoint.y);
        stepSize       = getGoldenSectionMin(stepSize, 1.0 - stepSize,
                                             currentPoint.x, currentGPoint.x, counter);
        currentPoint.x = computeNextPoint(currentPoint.x, stepSize, currentGPoint.x);
        currentPoint.y = computeNextPoint(currentPoint.y, stepSize, currentGPoint.y);

/*
        std::cout << "Step            : " << counter << std::endl;
        std::cout << "Size step       : " << stepSize << std::endl;
        std::cout << "Module Gradient : " << moduleGradient << std::endl;
        std::cout << "Gradient(x)     : " << currentGPoint.x << std::endl;
        std::cout << "Gradient(y)     : " << currentGPoint.y << std::endl;
        std::cout << "X               : " << currentPoint.x << std::endl;
        std::cout << "Y               : " << currentPoint.y << std::endl;
        std::cout << "Function(x,y)   : " << function(currentPoint.x, currentPoint.y);
        std::cout << std::endl << std::endl;
*/
    }
/*
    std::cout << "Start point: "
              << "(" << startPoint.x << ";" << startPoint.y << ")\t"
              << "x: " << currentPoint.x << "\t"
              << "y: " << currentPoint.y << "\t"
              << "f(x,y): " << function(currentPoint.x, currentPoint.y) << "\t"
              << "EPS: " << EPS << "\t"
              << "Step: " << counter
              << std::endl;
*/
    printf("Start point: (%.3f, %.3f)\tx: %.6f\ty: %.6f\tf(x,y) = %.3f\tEPS: %.8f\tStep %d\n",
           startPoint.x, startPoint.y, currentPoint.x, currentPoint.y,
           function(currentPoint.x, currentPoint.y), EPS, counter);
}

int main(int argc, char const *argv[])
{
    std::cout << "*** Function: ((x - 3) ^ 2 + 2 * (y - 4) ^ 2) ***" << std::endl;

    EPS = 1e-1;
    test(8.0, 8.0);

    EPS = 1e-3;
    test(8.0, 8.0);

    EPS = 1e-6;
    test(8.0, 8.0);

    EPS = 1e-8;
    test(8.0, 8.0);

    std::cout << std::endl;

    EPS = 1e-1;
    test(4.0, 4.0);

    EPS = 1e-3;
    test(4.0, 4.0);

    EPS = 1e-6;
    test(4.0, 4.0);

    EPS = 1e-8;
    test(4.0, 4.0);

    std::cout << std::endl;

    EPS = 1e-1;
    test(3.0, 4.0);

    EPS = 1e-3;
    test(3.0, 4.0);

    EPS = 1e-6;
    test(3.0, 4.0);

    EPS = 1e-8;
    test(3.0, 4.0);

    validation(function, std::make_pair(0.0, 5.0), std::make_pair(0.0, 5.0));

    return 0;
}
