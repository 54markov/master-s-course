#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

struct container
{
    int value;
    int pos;
};

container max(container a, container b)
{
    if (a.value > b.value)
        return a;

    return b;
}

static int skierTravel(std::vector<std::vector<int>> mountain)
{
    const int height = mountain.size();
    auto position = 0;

    auto score = mountain[0][0];

    for (auto i = 1; i < height; ++i)
    {
        container nextLeft = {
            .value = mountain[i][position],
            .pos = position
        };

        container nextRight = {
            .value = mountain[i][position + 1],
            .pos = position + 1
        };

        container rv = max(nextLeft, nextRight);

        position = rv.pos;
        score += rv.value;

#ifdef DEBUG
        std::cout << "pos    : " << rv.pos << std::endl;
        std::cout << "height : " << i << std::endl;
        std::cout << "value  : " << rv.value << std::endl;
#endif /* DEBUG */
    }

    return score;
}

static void parseArgs(char const *fileName, std::vector<std::vector<int>> &mountain)
{
    std::ifstream inFile(fileName);
    if (!inFile.is_open())
        throw "Can't open input file";

    std::string line;

    getline(inFile, line);

    int height = std::stoi(line);

    if ((height < 1) || (height > 100))
    {
        inFile.close();
        throw "Not valid argument";
    }

    for (auto i = 1; i < height + 1; ++i)
    {
        std::string line;

        getline(inFile, line);

        std::stringstream stream(line);

        int value;

        std::vector<int> newRow;

        while (stream >> value)
        {
            if ((value > 100) || (value < -100))
            {
                inFile.close();
                std::cerr << "Not valid number" << std::endl;
                exit(-1);                
            }
            newRow.push_back(value);
        }

        if ((int)newRow.size() != i)
        {
            inFile.close();
            std::cerr << "Not valid arguments number in line" << std::endl;
            exit(-1);
        }

        mountain.push_back(newRow);
    }

#ifdef DEBUG
    for_each(mountain.begin(), mountain.end(), [](std::vector<int> v)
    {
        for_each(v.begin(), v.end(), [](int v)
        {
            std::cout << v << " ";
        });
        std::cout << std::endl;
    });
#endif /* DEBUG */

    inFile.close();
}

static void parseArgsRunIt(char const *fileName)
{
    std::vector<std::vector<int>> mountain;

    parseArgs(fileName, mountain);

    auto answer = skierTravel(mountain);

    std::cout << answer << std::endl;

    std::ofstream outFile("slalom.out");

    if (!outFile.is_open())
        throw "Can't open input file";

    outFile << answer;

    outFile.close();
}

static void usage(char const *app)
{
    std::cout << "usage: " << app << " <file name>" << std::endl;
}

int main(int argc, char const *argv[])
{
    try
    {
        if (argc != 1)
        {
            parseArgsRunIt(argv[1]);
        }
        else
        {
            parseArgsRunIt("slalom.in");
            //usage(argv[0]);
        }
    }
    catch (std::string err)
    {
        std::cerr << err << std::endl;
        return -1;
    }

    return 0;
}
