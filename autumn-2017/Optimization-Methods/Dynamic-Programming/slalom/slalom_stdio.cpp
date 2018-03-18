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

static void parseArgs(std::vector<std::vector<int>> &mountain)
{
    int height = 0;

    std::cin >> height;

    if ((height < 1) || (height > 100))
        throw std::string("Not valid mountain height");

    for (auto i = 1; i < height + 1; ++i)
    {
        std::vector<int> newRow;
        for (auto j = 0; j < i; ++j)
        {
            int n;

            std::cin >> n;

            if ((n > 100) || (n < -100))
                throw std::string("Not valid number");

            newRow.push_back(n);
        }

        if ((int)newRow.size() != i)
        {
            throw std::string("Not valid arguments number in line");
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
}

int main(int argc, char const *argv[])
{
    std::vector<std::vector<int>> mountain;
    try
    {
        parseArgs(mountain);
    }
    catch (std::string err)
    {
        std::cerr << err << std::endl;
        exit(-1);
    }

    std::cout << skierTravel(mountain) << std::endl;

    return 0;
}
