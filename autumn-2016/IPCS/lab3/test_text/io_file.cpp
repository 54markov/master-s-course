#include "io_file.h"
#include "book_stack.h"
#include "bwt.h"

using namespace std;

void write_to_file(string buffer)
{
    ofstream myfile("compress_file", ios::out | ios::app | ios::binary);
    if (myfile.is_open()) {
        myfile << buffer;
    }
    myfile.close();
}

void read_from_file(const char *file)
{
    std::ifstream infile(file);

    MTF mtf;
    std::string key;
    std::string line;
    std::string filePayload;

    // Read whole file in one string
    while (std::getline(infile, line)) {
        filePayload += line;
    }
    infile.close();

    filePayload += "#";

    //cout << filePayload << endl;

    key = BWT(line);

    //cout << "key: " << key << endl;

    //reverseBWT(key);
    mtf.compress(mtf.encode(key));
}