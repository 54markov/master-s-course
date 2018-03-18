#include "functions.h"

void initAll(std::vector<Process*> &v,  std::vector<char> *a)
{
    Process *p = NULL;

    for (auto i = 0; i < 4; ++i) {
        p = new Process(i, TICKS, a);
        if (p) {
            v.push_back(p);
        } else {
            std::cerr << "Can't create Reader class" << endl;
            exit(1);
        }
    }
}

void freeAll(std::vector<Process*> &v)
{
    for (auto i = 0; i < 4; ++i) {
        Process *p = v[i];
        delete(p);
    }
    v.clear();  
}

void runSimulate(std::vector<Process*> &v, std::vector<char> &a)
{
    auto completeCounter = 0;
    std::srand(unsigned(std::time(0)));
    
    while (1) {
        int id = std::rand() % 4;

        int rc = v[id]->runProcess();

        if ((id == WRITER_1) && (rc == TICK_UNCOMPLETE)) {
            v[READER_1]->flushOut();
            v[READER_2]->flushOut();
            v[READER_3]->flushOut();
        }

        if (rc == TICK_COMPLETE) {
            completeCounter++;
        }

        if (completeCounter == 4)
        {
            printStat(v, a);
            break;
        }
    }
}

void printStat(std::vector<Process*> &v, std::vector<char> &a)
{
    std::cout << "GLOBAL ARRAY            : [ ";
    for (auto i : a) {
        std::cout << i << " ";
    }
    std::cout << "]" << endl;

    for (auto i = 0; i < 3; i++) {
        std::cout << "LOCAL ARRAY OF READER " << i << " : [ ";
        v[i]->printStat();
        std::cout << "]" << endl;
    }
}