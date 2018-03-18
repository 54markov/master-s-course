/*
 * A Dynamic Programming based solution for 0-1 Knapsack problem
 */

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

/*
 * Returns the maximum value that can be put in a knapsack of capacity W
 */
int knapsack(const std::vector<int> &wt, const int &W, const int &n)
{
    /*
     * wt - golden bars weights vector
     * W  - knapsack capacity
     * n  - amount of golden bars
     */

    int K[n + 1][W + 1];
 
    // Build table K[][] in bottom up manner
    for (auto i = 0; i <= n; ++i)
    {
        for (auto w = 0; w <= W; ++w)
        {
            if (i == 0 || w == 0)
            {
               K[i][w] = 0;
            }
            else if (wt[i - 1] <= w)
            {
                K[i][w] = std::max(wt[i - 1] + K[i - 1][w - wt[i - 1]],  K[i - 1][w]);
            }
            else
            {
                K[i][w] = K[i - 1][w];
            }
       }
   }
 
   return K[n][W];
}

static void parseArgs(char const *fileName, std::vector<int> &v, int &S, int &N)
{
    std::ifstream inFile(fileName);
    if (!inFile.is_open())
        throw std::string("Can't open input file");

    std::string line;

    getline(inFile, line);
    std::stringstream args(line);

    args >> S;
    args >> N;
/*
    std::cout << "knapsack capacity    : " << S << std::endl;
    std::cout << "amout of golden bars : " << N << std::endl;
*/
    if ((S < 1) || (S > 10000))
        throw std::string("Not valid S");

    if ((N < 1) || (N > 300))
        throw std::string("Not valid N");

    getline(inFile, line);
    std::stringstream bars(line);

    auto n = 0, cnt = 1;

    while (bars >> n)
    {
        if (cnt > N)
            throw std::string("overflow n");

        if (n < 0) 
            throw std::string("Not valid n");

        if (n > 100000)
            throw std::string("Not valid n");

        v.push_back(n);

        cnt++;
    }
/*
    std::for_each(v.begin(), v.end(), [](int val)
    {
        std::cout << val << " ";
    });
    std::cout << std::endl;
*/
    inFile.close();
}

static void parseArgsRunIt(char const *fileName)
{
    int S; // knapsack capacity
    int N; // amout of golden bars
    std::vector<int> v;

    parseArgs(fileName, v, N, S);

    std::ofstream outFile("knapsack.out");
    if (!outFile.is_open())
        throw std::string("Can't open input file");

    outFile << knapsack(v, N, S);

    outFile.close();
}

int main(int argc, char const *argv[])
{
    try
    {
        parseArgsRunIt("knapsack.in");
    }
    catch (std::string err)
    {
        std::cerr << err << std::endl;
        return -1;
    }

    return 0;
}
