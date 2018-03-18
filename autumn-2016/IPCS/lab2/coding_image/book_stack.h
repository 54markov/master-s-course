#ifndef _BOOK_STACK_H_
#define _BOOK_STACK_H_

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

#endif /* _BOOK_STACK_H_ */