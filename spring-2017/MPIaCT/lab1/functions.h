#ifndef _FUNCTIONS_
#define _FUNCTIONS_

#include <iostream>
#include <vector>

#include <random>
#include <ctime>

#include "classes.h"

void initAll(std::vector<Process*> &v,  std::vector<char> *a);
void freeAll(std::vector<Process*> &v);

void runSimulate(std::vector<Process*> &v, std::vector<char> &a);
void printStat(std::vector<Process*> &v, std::vector<char> &a);

#endif /* _FUNCTIONS_ */