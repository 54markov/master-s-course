#include <iostream>
#include <string>
#include <vector>

#include "classes.h"
#include "functions.h"

using namespace std;

int main(int argc, char const *argv[])
{
    vector<char>    vArray = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J' };
    vector<Process*> vProcess;

    initAll(vProcess, &vArray);

    runSimulate(vProcess, vArray);

    freeAll(vProcess);

    return 0;
}
