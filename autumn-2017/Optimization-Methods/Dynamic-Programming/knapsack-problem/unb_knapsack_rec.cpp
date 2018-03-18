#include <iostream>
#include <string>
#include <vector>
 
struct Item
{
    std::string name;
    double      value;
    double      weight;
    Item(std::string n, double v, double w) : name(n), value(v), weight(w) {}
};

std::vector<Item> items = {
    Item(std::string("item1"), 5.0,  3.0),
    Item(std::string("item2"), 9.0,  5.0),
    Item(std::string("item3"), 15.0, 8.0)
};
 
int n = items.size();
std::vector<int> count(items.size(), 0);
std::vector<int> best(items.size(), 0);
 
void knapsack(int i, double value, double weight, double &best_value)
{
    if (i == items.size())
    {
        if (value > best_value)
        {
            best_value = value;
            for (int j = 0; j < items.size(); j++)
            {
                best[j] = count[j];
            }
        }
        return;
    }
    
    int m = weight / items[i].weight;
    for (count[i] = m; count[i] >= 0; count[i]--)
    {
        double a = value + count[i] * items[i].value;
        double b = weight - count[i] * items[i].weight;
        knapsack(i + 1, a, b, best_value);
    }
}

int main(int argc, char const *argv[])
{
    double best_value = 0.0;
    
    knapsack(0, 0.0, 30.0, best_value);

    for (int i = 0; i < items.size(); i++)
    {
        std::cout << best[i] << " " << items[i].name << std::endl;
    }
    std::cout << "Best value: " << best_value << std::endl;

    return 0;
}
