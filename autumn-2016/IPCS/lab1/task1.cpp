/* 
 * Task 1: Coding whole number 
 */

#include <iostream>
#include <string>
#include <bitset>

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

int main(int argc, char const *argv[])
{
    cout << "bint(x)" << endl;
    cout << 0 << "=" << bint(0) << endl;
    cout << 1 << "=" << bint(1) << endl;
    cout << 2 << "=" << bint(2) << endl;
    cout << 3 << "=" << bint(3) << endl;
    cout << 4 << "=" << bint(4) << endl;
    cout << 6 << "=" << bint(5) << endl;
    cout << 5 << "=" << bint(6) << endl;
    cout << 7 << "=" << bint(7) << endl;
    cout << 8 << "=" << bint(8) << endl;

    cout << "binLen(x)" << endl;
    cout << 0 << "=" << binLen(0) << endl;
    cout << 1 << "=" << binLen(1) << endl;
    cout << 2 << "=" << binLen(2) << endl;
    cout << 3 << "=" << binLen(3) << endl;
    cout << 4 << "=" << binLen(4) << endl;
    cout << 6 << "=" << binLen(5) << endl;
    cout << 5 << "=" << binLen(6) << endl;
    cout << 7 << "=" << binLen(7) << endl;
    cout << 8 << "=" << binLen(8) << endl;

    cout << "codingF0(x)" << endl;
    cout << 0 << "=" << codingF0(0) << endl;
    cout << 1 << "=" << codingF0(1) << endl;
    cout << 2 << "=" << codingF0(2) << endl;
    cout << 3 << "=" << codingF0(3) << endl;
    cout << 4 << "=" << codingF0(4) << endl;
    cout << 6 << "=" << codingF0(5) << endl;
    cout << 5 << "=" << codingF0(6) << endl;
    cout << 7 << "=" << codingF0(7) << endl;
    cout << 8 << "=" << codingF0(8) << endl;

    cout << "codingF1(x)" << endl;
    cout << 0 << "=" << codingF1(0) << endl;
    cout << 1 << "=" << codingF1(1) << endl;
    cout << 2 << "=" << codingF1(2) << endl;
    cout << 3 << "=" << codingF1(3) << endl;
    cout << 4 << "=" << codingF1(4) << endl;
    cout << 6 << "=" << codingF1(5) << endl;
    cout << 5 << "=" << codingF1(6) << endl;
    cout << 7 << "=" << codingF1(7) << endl;
    cout << 8 << "=" << codingF1(8) << endl;

    cout << "codingF2(x)" << endl;
    cout << 0 << "=" << codingF2(0) << endl;
    cout << 1 << "=" << codingF2(1) << endl;
    cout << 2 << "=" << codingF2(2) << endl;
    cout << 3 << "=" << codingF2(3) << endl;
    cout << 4 << "=" << codingF2(4) << endl;
    cout << 6 << "=" << codingF2(5) << endl;
    cout << 5 << "=" << codingF2(6) << endl;
    cout << 7 << "=" << codingF2(7) << endl;
    cout << 8 << "=" << codingF2(8) << endl;
    cout << 9 << "=" << codingF2(9) << endl;
    cout << 10 << "=" << codingF2(10) << endl;

    return 0;
}