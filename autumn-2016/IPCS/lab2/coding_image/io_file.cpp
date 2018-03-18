#include "io_file.h"
#include "book_stack.h"

using namespace std;

void write_to_file(string byte)
{
    ofstream myfile("compressed_file", ios::out | ios::app | ios::binary);
    if (myfile.is_open()) {
        myfile << byte;
    }
    myfile.close();
}

void read_from_file(const char *file)
{
    std::ifstream infile(file);
    std::string line;
    std::string filePayload;
/*
    // Read from file, line by line
    while (std::getline(infile, line)) {
        MTF mtf;
        mtf.compress(mtf.encode(line));
    }
*/
    
    // Read whole file in one string
    while (std::getline(infile, line)) {
        filePayload += line;
    }
    infile.close();

    //cout << filePayload;

    MTF mtf;
    mtf.compress(mtf.encode(filePayload));
}