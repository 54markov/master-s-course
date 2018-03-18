/*
 * Unbounded Knapsack Problem:
 * - ANY item can be selected ANY number of times.
 */

#include <algorithm>
#include <iostream>
#include <vector>

struct Item
{
    int value;
    int weight;
    Item(int v, int w) : value(v), weight(w) { }
};

int picks[100][100];

/*
 * Returns the maximum value that can be put in a knapsack of capacity W
 */
auto unboundedKnapsack(const auto maxWeight, const std::vector<Item> items)
{
    /*
     * dp[i] is going to store maximum value, with knapsack capacity i
     */
    std::vector<int> dp;

    for (auto i = 0; i <= maxWeight; ++i)
    {
        dp.push_back(0);
        auto cnt = 0;
        auto icnt = 1;
        std::for_each(items.begin(), items.end(), [&](Item item)
        {
            if (item.weight <= i)
            {
                dp[i] = std::max(dp[i], dp[i - item.weight] + item.value);
                picks[cnt++][i] = dp[i];
            }
        });
    }

    for (auto j = 0; j <= maxWeight; ++j)
    {
        std::cout << j << "\t";
    }
    std::cout << std::endl;
    for (auto j = 0; j <= maxWeight; ++j)
    {
        std::cout << "-" << "\t";
    }
    std::cout << std::endl;

    for (auto i = 0; i < 3; ++i)
    {
        for (auto j = 0; j <= maxWeight; ++j)
        {
            std::cout << picks[i][j] << "\t";
        }
        std::cout << std::endl;
    }

    return *(dp.end() - 1);
}

int main(int argc, char const *argv[])
{
    const auto knapsackWeight = 10;

    const std::vector<Item> items = {
        Item(5, 3), Item(9, 5), Item(15, 8)
    };

    std::for_each(items.begin(), items.end(), [](Item i)
    {
        std::cout << "value : " << i.value << "\tweight : " << i.weight << std::endl;
    });

    std::cout << "Unbounded Knapsack Problem" << std::endl;
    std::cout << "Knapsack max weight   : " << knapsackWeight << std::endl;
    std::cout << "Knapsack max capacity : " << unboundedKnapsack(knapsackWeight, items) << std::endl;

    return 0;
}
