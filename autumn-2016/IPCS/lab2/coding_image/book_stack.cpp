/*
 * Move-To-Front
 */

#include "book_stack.h"
#include "code_number.h"
#include "io_file.h"

using namespace std;

char set_bit(int num, int pos) { return (num |= 1 << pos); }
char clr_bit(int num, int pos){ return (num &= ~(1 << pos)); }

void MTF::moveToFront(int pos)
{
    char temp = symbolTable_[pos];

    for (auto i = pos - 1; i >= 0; i--) {
        symbolTable_[i + 1] = symbolTable_[i];
    }
    symbolTable_[0] = temp;
}
     
void MTF::fillSymbolTable()
{
    for (auto x = 0; x < 26; ++x) {
        symbolTable_[x] = x + 'a';
    }
}

string MTF::encode(string str)
{
    vector<int> output;

    /* Filling the alphabet */
    fillSymbolTable();

    /* Loop by intput string */
    for (auto it = str.begin(); it != str.end(); ++it) {
        /* Search str[i] in alphabet */
        for (auto i = 0; i < 26; i++) {
            if (*it == symbolTable_[i]) {   
                output.push_back(i); // save in vector, position number
                moveToFront(i);      // move alphabet
                break;
            }
        }
    }

    string r;

    /* Loop by saving vector */
    for (auto it = output.begin(); it != output.end(); it++) {
        ostringstream ss;
        /* Convert vector<int> to strring */
        ss << *it;
        r += ss.str() + " ";
    }
    return r;
}
     
string MTF::decode(string str)
{
    istringstream iss(str);
    vector<int> output;

    fillSymbolTable();
            
    copy(istream_iterator<int>(iss), istream_iterator<int>(), back_inserter<vector<int>>(output));
            
    string r;

    for (auto it = output.begin(); it != output.end(); it++) {
        r.append(1, symbolTable_[*it]);
        moveToFront(*it);
    }

    return r;
}

void MTF::compress(string str)
{
    istringstream iss(str);
    vector<int> output;

    copy(istream_iterator<int>(iss), istream_iterator<int>(), back_inserter<vector<int>>(output));
            
    for (auto it = output.begin(); it != output.end(); it++) {
        write_to_file(codingF2(*it));
    }
}
