#include "classes.h"

Process::Process() {};
Process::Process(std::string _procId) : procId_(_procId), procTick_(0) {};
Process::~Process() {};

int Process::runProc()
{
    if (procTick_ == TICKS) {
        procTick_++;
        return TICK_COMPLETE;
    } else if (procTick_ > TICKS) {
        return TICK_END;
    } else {
        procTick_++;
        return TICK_UNCOMPLETE;
    }
}

std::string Process::getProcId()
{
    return procId_;
}