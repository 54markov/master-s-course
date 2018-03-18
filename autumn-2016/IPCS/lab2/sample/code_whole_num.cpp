/* 
 * Task 1: Coding whole number 
 */

#include "code_whole_num.h"

using namespace std;

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