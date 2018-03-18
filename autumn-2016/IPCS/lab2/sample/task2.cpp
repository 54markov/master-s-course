/*
 * Move to the front
 */

#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

#include "code_whole_num.h"
 
using namespace std;
 
class MTF
{
    private:
        char symbolTable_[26];

        void moveToFront(int pos) {
            char temp = symbolTable_[pos];
            for (auto i = pos - 1; i >= 0; i--) {
                symbolTable_[i + 1] = symbolTable_[i];
            }
            symbolTable_[0] = temp;
        }
     
        void fillSymbolTable() {
            for (auto x = 0; x < 26; ++x) {
                symbolTable_[x] = x + 'a';
            }
        }

    public:
        /* coding string */
        string encode(string str) {
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
     
        string decode(string str) {
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

        void compress(string str) {
            istringstream iss(str);
            vector<int> output;

            copy(istream_iterator<int>(iss), istream_iterator<int>(), back_inserter<vector<int>>(output));
            for (auto it = output.begin(); it != output.end(); it++) {
                cout << *it << "\t: " << codingF2(*it) << endl;
            }
        }
};

int main(int argc, char const *argv[])
{
    MTF mtf;

    string str[] = { "broood", "bananaaa", "hiphophiphop", "baadaade" };
    
    for (auto i = 0; i < 4; i++) {
        string a = str[i];
        cout << "-> original = " << str[i] << "\n-> encoded  = ";
        a = mtf.encode(a);
        cout << a << "\n-> decoded  = " << mtf.decode(a) << endl << endl;
    }

    cout << "--Coding and compressing--" << endl;

    string file("file");
    cout << "-> original = " << file << "\n-> encoded  = ";
    cout << mtf.encode(file) << endl;
    mtf.compress(mtf.encode(file));

    return 0;
}