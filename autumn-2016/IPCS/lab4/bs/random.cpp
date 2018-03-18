#include <random>
#include <iostream>

#include <sstream>
#include <string>
#include <fstream>

#include <bitset>

using namespace std;

int get_bit(int byte, int bit)
{
    return((byte >> 32 - bit) & 1);
}

char set_bit(char num, int pos) { return (num |= 1 << pos); }
char clr_bit(char num, int pos) { return (num &= ~(1 << pos)); }

int main()
{
	ofstream myfile("rand_file", ios::out | ios::app | ios::binary);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    if (myfile.is_open()) {
        for (int n = 0; n < 1000; ++n) {
            char ch = 0;
            int num = dis(gen);

            for (auto i = 0, k = 24; i < 8; i++, k++) {
                if (get_bit(num, k) == 0) {
                    ch = clr_bit(ch, i);
                } else {
                    ch = set_bit(ch, i);
                }
            }
            myfile << ch;
    	}
    }
 	myfile.close();   
    return 0;
}