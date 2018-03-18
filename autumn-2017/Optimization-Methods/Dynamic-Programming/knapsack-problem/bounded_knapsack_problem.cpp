/*
 * Bounded Knapsack Problem:
 * - ANY item can be selected LIMITED times.
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
 
/*
 * Returns the maximum value that can be put in a knapsack of capacity W
 */
int boundedKnapsack(const int maxWeight, const std::vector<Item> items)
{
    std::vector<std::vector<int>> K;
 
    for (auto i = 0; i <= static_cast<int>(items.size()); ++i)
    {
        K.push_back(std::vector<int>(maxWeight + 1, 0));

        for (auto w = 0; w <= maxWeight; ++w)
        {
            if (!i || !w)
            {
               K[i][w] = 0;
            }
            else if (items[i - 1].weight <= w)
            {
                auto prevValue  = items[i - 1].value;
                auto prevWeight = items[i - 1].weight;
                K[i][w] = std::max(prevValue + K[i - 1][w - prevWeight], K[i - 1][w]);
            }
            else
            {
                K[i][w] = K[i - 1][w];
            }
       }
   }

   return *((K.end() - 1)->end() - 1);
}
 
int main(int argc, char const *argv[])
{
    const auto knapsackWeight = 10;

    const std::vector<Item> items = {
        Item(1, 1), Item(4, 1), Item(8, 1)
    };

    std::cout << "Bounded Knapsack Problem" << std::endl;
    std::cout << "Knapsack max weight   : " << knapsackWeight << std::endl;
    std::cout << "Knapsack max capacity : " << boundedKnapsack(knapsackWeight, items) << std::endl;

    return 0;
}
