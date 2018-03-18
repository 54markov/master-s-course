#ifndef _CODE_WHOLE_NUM_
#define _CODE_WHOLE_NUM_

#include <iostream>
#include <string>
#include <bitset>

#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

using namespace std;

class MTF
{
    private:
        char symbolTable_[26];

        void moveToFront(int pos);
        void fillSymbolTable();

    public:
        /* coding string */
        string encode(string str);
        string decode(string str);
        void compress(string str);
};

string bint(int x);
int binLen(int x);
string codingF0(int wholeNumber);
string codingF1(int x);
string codingF2(int x);

#endif /* _CODE_WHOLE_NUM_ */