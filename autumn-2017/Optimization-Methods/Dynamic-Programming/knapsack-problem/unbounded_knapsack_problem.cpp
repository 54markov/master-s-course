/*
 * Unbounded Knapsack Problem:
 * - ANY item can be selected ANY number of times.
 */

#include <iostream>
#include <vector>
#include <string>

struct Item
{
    std::string name;
    int         value;
    int         weight;

    Item(std::string n, int v, int w) : name(n), value(v), weight(w) { }
};

/*
 * Print out the actual weights and their no. of instances used
 * to fill the knapsack
 */
void instancesUsed(const std::vector<Item> items,
                   const int               capacity,
                   std::vector<int>        usedItems)
{
    int size = items.size();
    int k = capacity;
    std::vector<int> instances(size, 0);

    /*
     * Compute the no. of instances used for each selected item(weight)
     */
    while (k >= 0)
    {
        auto x = usedItems[k];
        if (x == -1)
        {
            break;
        }
        
        instances[x] += 1;
        k -= items[usedItems[k]].weight;
    }

    std::cout<< "Instances used:" << std::endl;
    for (auto i = 0; i < size; i++)
    {
        std::cout << items[i].name; 
        std::cout << " weight(" << items[i].weight << ")";
        std::cout << " value("  << items[i].value  << ")";
        std::cout << "\t- "      << instances[i]    << std::endl;
    }
}

/* 
 * Given n items, each having weight w(i) and value v(i) and a 
 * knapsack of weight 'W', find the maximum value that can be 
 * accommodated using 1 or more instances of given weights.
 * Suppose knapsack[j] -> max value that can be fitted in a knapsack
 *                        of weight 'j'
 * Then, knapsack[W] -> required answer  
 * Standard Recursive solution :-
 * knapsack[j] = max(knapsack[j-1], {knapsack[j-w(i)]+v(i) for i = 0...n-1})
 */
int unboundedKnapsack(const std::vector<Item> items,
                      const int               capacity,
                      std::vector<int>        &usedItems)
{
    int n = items.size();

    /*
     * Temporary array where index 'j' denotes max value that can be fitted
     * in a knapsack of weight 'j'
     */
    std::vector<int> knapsack(capacity + 1, 0);

    usedItems[0] = -1;

    for (auto j = 1; j <= capacity; ++j)
    {
        usedItems[j] = usedItems[j - 1];

        /*
         * As per our recursive formula,
         * iterate over all weights w(0)...w(n-1)
         * and find max value that can be fitted in knapsack of weight 'j'
         */
        auto max = knapsack[j - 1];
        for (auto i = 0; i < n; ++i)
        {
            auto x = j - items[i].weight;
            if (x >= 0 && (knapsack[x] + items[i].value) > max)
            {
                max = knapsack[x] + items[i].value;
                usedItems[j] = i;
            }
            knapsack[j] = max;
        }
   }

   return knapsack[capacity];
}

int main(int argc, char const *argv[])
{
    /*
     * Items of the knapsack
     */
    const std::vector<Item> items = {
        Item(std::string("Item1"), 5,  3),
        Item(std::string("Item2"), 9,  5),
        Item(std::string("Item3"), 15, 8)
    };

    /*
     * Capacity of the knapsack
     */
    const int capacity = 30;

    /*
     * Stores the items and the no. of instances of each item
     * used to fill the knapsack
     */
    std::vector<int> usedItems(capacity + 1, 0);
    auto maxValue = unboundedKnapsack(items, capacity, usedItems);

    std::cout << "Unbounded Knapsack Problem" << std::endl;
    std::cout << "Knapsack max weight              : " << capacity << std::endl;
    std::cout << "Maximum value that can be fitted : " << maxValue << std::endl;

    instancesUsed(items, capacity, usedItems);

    return 0;
}
