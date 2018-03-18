#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <string.h>

#include "bwt.h"

using namespace std;

int PARAM = 1;

bool pairCompare(const std::pair<string, string>& firstElem, const std::pair<string, string>& secondElem) 
{
    string a = firstElem.first;
    string b = secondElem.first;

    a.erase(a.end()-1);
    b.erase(b.end()-1);

    reverse(a.begin(), a.end());
    reverse(b.begin(), b.end());

    return a < b;
}

string BWT(string str)
{
    string cp = str;
    vector<string> table_shuffle;
    vector<pair<string,string>> table_sort;

    table_shuffle.push_back(str);
    table_sort.push_back(make_pair(str,str));

    for (auto i = 0; i < str.size() - 1; ++i) {
        char ch = str[0];
        for (auto j = 1; j < str.size(); ++j) {
            str[j - 1] = str[j];
        }
        str[str.size() - 1] = ch;

        table_shuffle.push_back(str);
        table_sort.push_back(make_pair(str,str));
    }

    sort(table_sort.begin(), table_sort.end(), pairCompare);

    auto i = 0;
    for (auto j : table_sort) {
        //cout << table_shuffle[i] << " -- " << j.second << endl;
        i++;

        if (cp.compare(j.second) == 0) {
            PARAM = i - 1;
        }
    }

    string key;
    for (auto j : table_sort) {
        key += j.second[j.second.size() - 1];
    }

    return key;
}

void reverseBWT(string key)
{
    vector<string> table;

    for (auto i = 0; i < key.size(); ++i) {
        string str;
        str += key[i];       
        table.push_back(str);
    }
/*
    cout << "Input: " << endl;
    for (auto i = 0; i < table.size(); ++i) {
        cout << table[i] << endl;
    }
*/
    vector<string> table1 = table;

    for (auto i = 1; i < key.size(); ++i) {

        sort(table1.begin(), table1.end());

        for (auto j = 0; j < table.size(); ++j) {
            table1[j] = table[j] + table1[j];
        }
/*
        cout << endl;
        for (auto i = 0; i < table.size(); ++i) {
            cout << table1[i] << endl;
        }
        cout << endl;
*/
    }

    sort(table1.begin(), table1.end());
/*
    cout << endl;
    for (auto i = 0; i < table.size(); ++i) {
        cout << table1[i] << endl;
    }
    cout << endl;
*/

    cout << "Output: ";
    reverse(table1[PARAM].begin(), table1[PARAM].end());

    table1[PARAM].erase(table1[PARAM].begin());
    cout << table1[PARAM] << endl;
}