#include "Field.h"
#include "Colors.h"

#include <iostream>
#include <vector>
#include <random>
#include <climits>

#include <algorithm>
#include <cassert>
#include <iterator>

using namespace std;

Field::Field(int x, int y)
{
    for (auto i = 0; i < x; ++i) {
        std::vector<char> newV(y);
        for (auto j = 0; j < y; ++j) {
            newV[j] = '.';
        }
        this->field_.push_back(newV);
    }
}

Field::~Field() {}

void Field::printField()
{
    for (auto i : field_) {
        for (auto j: i) {
            //std::cout << "[" << j << "]";
            if (j != '.') {
                std::cout << KRED << j << RST;
            } else {
                std::cout << j;
            }
        }
        std::cout << std::endl;
    }
}

int Field::setWayPoint(int x, int y)
{
    static char wayPoint = 'A';
    try {
        if ((x > 100) || (x < 0))
            throw std::string("ERROR: not valid X");
        if ((y > 100) || (y < 0))
            throw std::string("ERROR: not valid Y");
    } catch (std::string err) {
        std::cerr << __FUNCTION__ << ":";
        std::cerr << err << std::endl;
        exit(1);
    }

    if (this->field_[x][y] != '.') {
        return 1;
    } else {
        this->field_[x][y] = wayPoint++;
        wayPoints_.push_back(std::make_pair(x,y));
        return 0;
    }
}

void Field::generateWayPoints(int wayPoints)
{
    for (auto i = 0; i < wayPoints; ) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 99);

        int x = dis(gen);
        int y = dis(gen);
/*
        std::cout << x << ":" << y << std::endl;
*/
        if (this->setWayPoint(x, y) == 0) {
            i++;
        }
    }
}

void Field::getPathBetweenWayPoints()
{
/*
    std::vector<std::pair<int, int> > wayPoints;
    int x = 0;
    for (auto i : field_) {
        int y = 0;
        for (auto j: i) {
            if (j != '.') {
                wayPoints.push_back(std::make_pair(x,y));
            }
            y++;
        }
        x++;
    }
*/
    this->permutation(wayPoints_);
}

int Field::getPathBetweenWayPoints_(std::pair<int, int> a, std::pair<int, int> b)
{
    auto x = abs(a.first - b.first);
    auto y = abs(a.second - b.second);

    auto straight = x + y /*- 1*/;
    auto diagonally = abs(x - y) + ((x > y) ? y : x);
/*
    std::cout << std::endl;
    std::cout << KGRN << "abs(x - y) = " << abs(x - y) << RST << std::endl;
    std::cout << KGRN << "x          = " << x << RST << std::endl;
    std::cout << KGRN << "y          = " << y << RST << std::endl;
    std::cout << KGRN << "diagonally = " << diagonally << RST << std::endl;


    std::cout << KYEL << ((straight > diagonally) ? diagonally : straight) << RST << std::endl;

    std::cout << KYEL << "x=" << x << "/ y =" << y << RST << std::endl;
    std::cout << KYEL << "straight = " << straight << "/ diagonally =" << diagonally << RST << std::endl;
*/
    return ((straight > diagonally) ? diagonally : straight);
}

//template <typename T>

//void Field::permutation(std::vector<T> v)
void Field::permutation(std::vector<std::pair<int, int> > vecPair)
{
    std::vector<int> v;
    auto minPath = INT_MAX;

    // Generate permutation vector
    for (auto i = 1; i < (int)vecPair.size(); ++i) {
        v.push_back(i);
    }

    //std::cout << KGRN << "v.size() = " << v.size() << RST << std::endl;


    std::sort(v.begin(), v.end());
    
    auto commonPath = 0;

    do {
        commonPath += getPathBetweenWayPoints_(vecPair[0], vecPair[v[0]]);

        for (auto i = 0; i < (int)v.size() - 1; ++i) {
            commonPath += getPathBetweenWayPoints_(vecPair[v[i]], vecPair[v[i+1]]);
        }
        commonPath += getPathBetweenWayPoints_(vecPair[v.size() - 1], vecPair[0]);
        //std::cout << KYEL << commonPath << RST << std::endl;
        
        if (minPath > commonPath) {
            minPath = commonPath;
        }
    } while (std::next_permutation(v.begin(), v.end()));
    std::cout << KBLU << minPath << RST << std::endl;
}
/*
void Field::permutations(std::vector<std::pair<int, int> > arr, int size)
{
    auto minPath = INT_MAX;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size - 1; j++) {
            auto commonPath = 0;
            for (int k = 0; k < size - 1; k++) {
                commonPath += getPathBetweenWayPoints_(arr[k], arr[k+1]);
            }
            if (minPath > commonPath) {
                minPath = commonPath;
            }
            //std::cout << KYEL << commonPath << RST << std::endl;
            std::pair<int, int> tmp = arr[j];
            arr[j]   = arr[j+1];
            arr[j+1] = tmp;
        }
    }
    std::cout << KBLU << minPath << RST << std::endl;
}
*/