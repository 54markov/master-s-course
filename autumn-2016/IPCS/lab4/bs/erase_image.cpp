#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

using namespace std;

void writeToFile(string buffer)
{
    ofstream myfile("mountains", ios::out | ios::app | ios::binary);
    if (myfile.is_open()) {
        myfile << buffer;
    }
    myfile.close();
}

void readFromFile(const char *file)
{
    std::ifstream infile(file);
    std::string line;
    std::string filePayload;

    // Read whole file in one string
    while (std::getline(infile, line)) {
        filePayload += line;
    }
    infile.close();

    filePayload.erase(filePayload.begin(), filePayload.begin() + 54); 

    writeToFile(filePayload);
}

int main(int argc, char const *argv[])
{
    readFromFile("mountains.jpg");
    return 0;
}