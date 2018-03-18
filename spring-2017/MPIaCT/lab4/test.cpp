
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <vector>

#include <sys/time.h>
#include <stdio.h>
 
template <typename T>
void Permutation(std::vector<T> v)
{
    std::sort(v.begin(), v.end());
    do {
        std::cout << "1 ";
        //std::copy(v.begin(), v.end()/*, std::ostream_iterator<T>(std::cout, " ")*/);

        for (auto i : v) {
            std::cout << i << " ";
        }
        std::cout << "[last = " << v[v.size() - 1] << "] ";
        std::cout << "1";
        std::cout << std::endl;
    } while (std::next_permutation(v.begin(), v.end()));
}

/*
template <typename T>
void Combination(const std::vector<T>& v, std::size_t count)
{
    assert(count <= v.size());
    std::vector<bool> bitset(v.size() - count, 0);
    bitset.resize(v.size(), 1);
 
    do {
        for (std::size_t i = 0; i != v.size(); ++i) {
            if (bitset[i]) {
                std::cout << v[i] << " ";
            }
        }
        std::cout << std::endl;
    } while (std::next_permutation(bitset.begin(), bitset.end()));
}
 
bool increase(std::vector<bool>& bs)
{
    for (std::size_t i = 0; i != bs.size(); ++i) {
        bs[i] = !bs[i];
        if (bs[i] == true) {
            return true;
        }
    }
    return false; // overflow
}
 
template <typename T>
void PowerSet(const std::vector<T>& v)
{
    std::vector<bool> bitset(v.size());
 
    do {
        for (std::size_t i = 0; i != v.size(); ++i) {
            if (bitset[i]) {
                std::cout << v[i] << " ";
            }
        }
        std::cout << std::endl;
    } while (increase(bitset));
}
*/
double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}
 
int main()
{
    auto t = wtime();
    for (auto j = 0; j < 10; j++) {
        std::vector<int> vc { 2, 3, 4 };
        std::cout << "\n------4.PERMUTATION----------\n";
        Permutation(vc);
    }
    t = wtime() - t;
    printf("4. Elapsed time: %.6f sec.\n", t / 10);

/*
    t = wtime();
    for (auto j = 0; j < 10; j++) {
        std::vector<char> vc { 'A', 'B', 'C', 'D', 'E' };
        std::vector<char> path;
        std::vector<bool> visited(5, false);
        std::cout << "\n------5.PERMUTATION----------\n";
        Permutation(vc);    
    }
    t = wtime() - t;
    printf("5. Elapsed time: %.6f sec.\n", t / 10);


    t = wtime();
    for (auto j = 0; j < 10; j++) {
        std::vector<char> vc { 'A', 'B', 'C', 'D', 'E', 'F' };
        std::vector<char> path;
        std::vector<bool> visited(6, false);
        std::cout << "\n------6.PERMUTATION----------\n";
        Permutation(vc);    
    }
    t = wtime() - t;
    printf("6. Elapsed time: %.6f sec.\n", t / 10);

    t = wtime();
    for (auto j = 0; j < 10; j++) {
        std::vector<char> vc{ 'A', 'B', 'C', 'D', 'E', 'F', 'G' };
        std::vector<char> path;
        std::vector<bool> visited(7, false);
        std::cout << "\n------7.PERMUTATION----------\n";
        Permutation(vc);    
    }
    t = wtime() - t;
    printf("7. Elapsed time: %.6f sec.\n", t / 10);

    t = wtime();
    for (auto j = 0; j < 10; j++) {
        std::vector<char> vc{ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H' };
        //std::vector<char> path;
        //std::vector<bool> visited(8, false);
        std::cout << "\n------8.PERMUTATION----------\n";
        Permutation(vc);    
    }
    t = wtime() - t;
    printf("8. Elapsed time: %.6f sec.\n", t / 10);
*/

/*    
    std::cout << "\n------COMBINATION----------\n";
    Combination(vc, 3);
    std::cout << "\n------POWERSET-------------\n";
    PowerSet(vc);
*/
    return 0;
}