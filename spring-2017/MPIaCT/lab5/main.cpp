#include <iostream>
#include <vector>

#include <random>
#include <climits>

#include <algorithm>
#include <cassert>
#include <iterator>

#include <sys/time.h>
#include <stdio.h>

using namespace std;

/*****************************************************************************/
/* Prototypes                                                                */
/*****************************************************************************/
double wtime();
void createTools(int tools);
void permutation(std::vector<std::pair<int, int>> v);
void printTools(const std::vector<std::pair<int, int>> &v);


double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

void printTools(const std::vector<std::pair<int, int>> &v)
{
    cout << v.size() << " tools" << endl;
    for (auto i : v) {
        cout << i.first;
        if (i.first >= 10)
            cout << " : ";
        else
            cout << "  : ";
        cout << i.second << endl;
    }
    cout << endl;
}

void createTools(int tools)
{
#ifdef DEBUG
    cout << __FUNCTION__ << "()" << endl;
#endif

    std::vector<std::pair<int, int>> vecTools;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(1, 15);

    for (auto i = 0; i < tools; ++i) {
        auto timeBench1 = dist(gen);
        auto timeBench2 = dist(gen);
        vecTools.push_back(std::make_pair(timeBench1, timeBench2));
    }
#ifdef DEBUG
    printTools(vecTools);
#endif /* DEBUG */

    permutation(vecTools);
}

void permutation(std::vector<std::pair<int, int>> v)
{
#ifdef DEBUG
    cout << __FUNCTION__ << "()" << endl;
#endif
    int size = v.size();
    auto minTime = INT_MAX;

    std::sort(v.begin(), v.end());

    do {
#ifdef DEBUG
        printTools(v);
#endif /* DEBUG */
        auto permutationTime = 0;
        permutationTime = v[0].first + v[0].second;

        for (auto i = 1; i < size; i++) {
            if (v[i].first > v[i-1].second) {
                permutationTime += v[i].first - v[i-1].second;  
            } else {
                permutationTime += v[i].second;
            }
        }
#ifdef DEBUG
        cout << permutationTime << endl;
#endif /* DEBUG */
        if (minTime > permutationTime) {
            minTime = permutationTime;
        }
    } while (std::next_permutation(v.begin(), v.end()));

    cout << "Minimum time is: " << minTime << endl;
}

int main(int argc, char const *argv[])
{
    for (auto i = 5; i < 13; ++i) {
        auto t = wtime();
        createTools(i);
        t = wtime() - t;
        printf("Elapsed time (%d): %.6f sec.\n", i, t);
    }
    return 0;
}