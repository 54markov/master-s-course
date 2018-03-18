#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>

static inline std::pair<int, int> maxValueIndex(std::pair<int, int> &a,
                                                std::pair<int, int> &b)
{
    if (a.second > b.second)
        return a;
    else
        return b;
}

static int ladderClimb(std::vector<int> value)
{
    std::vector<int> sum;
    sum.push_back(value[0]);
    sum.push_back(value[1]);

    for (auto i = 2; i < static_cast<int>(value.size()); i++)
    {
        std::pair<int, int> sumPrev(i - 1, sum[i - 1]);
        std::pair<int, int> sumPrevPrev(i - 2, sum[i - 2]);
        
        auto result = maxValueIndex(sumPrev, sumPrevPrev); 

        sum.push_back(result.second + value[i]);
    }

    auto answer = 0;

    if (static_cast<int>(value.size()) < 3)
    {
        std::for_each(sum.begin(), sum.end(), [&](int value)
        {
            answer += value;
        });
    }
    else
    {
        answer = sum[sum.size() -1];
    }

    std::cout << "Ladder Climb Score: " << answer << std::endl;

    return answer;
}

static void parseArgs(char const *fileName, std::vector<int> &v, int &size)
{
    std::ifstream inFile(fileName);
    if (!inFile.is_open())
    {
        std::cerr << "Can't open input file" << std::endl;
        exit(-1);
    }
    
    std::string line;
    
    getline(inFile, line);
    
    size = std::stoi(line);

    if ((size < 1) || (size > 100))
    {
        std::cerr << "Not valid argument" << std::endl;
        exit(-1);
    }
    
    getline(inFile, line);

    std::stringstream stream(line);

    for (auto i = 0; i < size; i++) {
        int n;
        stream >> n;
        v.push_back(n);
    }

    inFile.close();
}

static void parseArgsRunIt(char const *fileName)
{
    int size = 0;
    std::vector<int> value;

    parseArgs(fileName, value, size);

    std::ofstream outFile("ladder.out");

    if (!outFile.is_open())
        throw "Can't open input file";

    outFile << ladderClimb(value);;

    outFile.close();
}

int main(int argc, char const *argv[])
{
    if (argc != 1)
    {
        parseArgsRunIt(argv[1]);
    }
    else
    {
        parseArgsRunIt("ladder.in");
    }
    return 0;
}
