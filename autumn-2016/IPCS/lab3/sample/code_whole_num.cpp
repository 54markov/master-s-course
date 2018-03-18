/* 
 * Task 1: Coding whole number 
 */

#include "code_whole_num.h"

using namespace std;

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
        cout << *it << "\t: " << codingF2(*it) << endl;
    }
}

string bint(int x)
{
    if (x < 1)
        return "";

    string s = bitset<64>(x).to_string();

    auto offset = s.find("1") + 1;
    string str;

    for (auto i = offset; i < 64; ++i) {
        str += s[i];
    }
    return str;
}

int binLen(int x)
{
    if (x < 1)
        return 1;

    string s = bitset<64>(x).to_string();

    auto offset = s.find("1");

    return 64 - offset;
}

string codingF0(int wholeNumber)
{
    if (wholeNumber == 0) {
        return "1"; 
    }

    string codeF0;

    for (auto i = 0; i < wholeNumber; ++i) {
        codeF0 += "0";
    }

    codeF0 += "1";

    return codeF0;
}

string codingF1(int x)
{
    if (x == 0) {
        return "1";
    }

    string code = codingF0(binLen(x));
    
    code += bint(x);

    return code;
}

string codingF2(int x)
{
    if (x == 0) {
        return "1";
    }

    if (x == 1) {
        return "01";
    }

    string code = codingF1(binLen(x));

    code += bint(x);

    return code;
}