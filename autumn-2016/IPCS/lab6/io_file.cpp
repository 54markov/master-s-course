#include "io_file.h"
#include "book_stack.h"
#include "bwt.h"

using namespace std;

std::string file_name;

void set_file_name(const char* file)
{
    file_name = file;

    file_name.erase(file_name.begin(), file_name.begin()+ 5);
    
    file_name = "result/" + file_name;
    file_name += ".compress_file";

    cout << file_name << endl;
}

void write_to_file(string buffer)
{
    ofstream myfile(file_name, ios::out | ios::app | ios::binary);
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