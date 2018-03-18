#include <iostream>
#include <string>
#include <vector>

#include "classes.h"
#include "functions.h"

using namespace std;

int main(int argc, char const *argv[])
{
	int allR = 0;
    int allW = 0;
    int completeR = 0;
    int completeW = 0;

    runSimulate(allR, allW, completeR, completeW);
    showStat(allR, allW, completeR, completeW);

    return 0;
}
