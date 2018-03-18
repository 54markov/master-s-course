#ifndef _CLASSES_
#define _CLASSES_

#include <iostream>
#include <vector>

#define TICKS 10

enum {
    TICK_COMPLETE   = 0,
    TICK_UNCOMPLETE = 1,
    TICK_END        = 2
};

using namespace std;

class Process
{
    private:
        std::string procId_;
        int         procTick_;

    public:
        Process();
        Process(std::string _procId);
        ~Process();
        int runProc();
        std::string getProcId();
};

#endif /* _CLASSES_ */
