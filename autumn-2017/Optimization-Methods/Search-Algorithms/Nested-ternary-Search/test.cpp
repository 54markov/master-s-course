#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>

struct container
{
    double x;
    double y;
    double f;
};

const double goldenSection = 1.61803398875;

double function2(double x, double y)
{
    return (std::pow(x - std::pow(y, 2), 2) + 0.1 * std::pow((x - 0.7), 2));
}

container validation(double (* f)(double, double),
                     std::pair<double,double> &x,
                     std::pair<double,double> &y)
{
    double step = 0.01;
    std::vector<container> v;

    for (double i = x.first; i < x.second; i += step)
    {
        for (double j = y.first; j < y.second; j += step)
        {
            container t = {
                .x = j,
                .y = i,
                .f = f(j, i)
            };

            v.push_back(t);
        }
    }

    std::sort(v.begin(), v.end(), [](const container &a,
                                     const container &b)
    {
        return a.f < b.f; 
    });

    auto rv = *(v.begin());

    return rv;
}

void nestedTernarySearch(double (* f)(double, double),
                         std::pair<double,double> &x,
                         std::pair<double,double> &y,
                         double                   eps)
{
    auto rv_v = validation(f, x, y); // return value validation
    int counter = 0;
    double step = 0.01;

    std::vector<container> v;

    // Search function minimum
    for (double i = x.first; i < x.second; i += step)
    {
        double left      = y.first;
        double right     = y.second;
        double prevValue = 0.0;

        while (right - left > eps)
        {
            double a = right - ((right - left) / goldenSection);
            double b = left  + ((right - left) / goldenSection);
            bool firstInvoke = true;

            if (firstInvoke)
            {

                double Fa = f(i, a);
                double Fb = f(i, b);

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
                double Fi = f(i, b);
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

        container t = {
            .x = i,
            .y = ((left + right) / 2.0),
            .f = f(i, ((left + right) / 2.0))
        };

        v.push_back(t);
    }

    std::sort(v.begin(), v.end(), [](const container &a,
                                     const container &b)
    {
        return a.f < b.f; 
    });

    auto rv_ts = *v.begin();

    std::cout << "Fmin (nested teranry search) : " << rv_ts.f;
    std::cout << " in point (" << rv_ts.x << ", " <<  rv_ts.y << ")" << std::endl;
    std::cout << "Fmin (naive)                 : " << rv_v.f;
    std::cout << " in point (" << rv_v.x << ", " <<  rv_v.y << ")" << std::endl;
    std::cout << "*Steps                       : " << counter << std::endl;
    std::cout << "*Epsilon                     : " << eps << std::endl;
    std::cout << std::endl;
}

int main(int argc, char const *argv[])
{
    std::pair<double,double> x2(-5.0, 5.0);
    std::pair<double,double> y2(-5.0, 5.0);

    std::cout << "\n ** Function : (x - y ^ 2) ^ 2 + 0.1 * (x - 0.7) ^ 2; [-5, 5], [-5, 5] **\n" << std::endl;
    nestedTernarySearch(function2, x2, y2, 0.01);
    nestedTernarySearch(function2, x2, y2, 0.0001);
    nestedTernarySearch(function2, x2, y2, 0.000001);

    return 0;
}
