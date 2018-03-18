#include "timer.h"

#include <iostream>

#include <signal.h>
#include <sys/time.h>

#include <unistd.h>

using namespace std;

Timer::Timer(int timeToLive)
{
    struct itimerval nval, oval;

    nval.it_interval.tv_sec  = timeToLive; // interval 
    nval.it_interval.tv_usec = 0;
    nval.it_value.tv_sec     = timeToLive; // time until next expiration
    nval.it_value.tv_usec    = 0;

    setitimer(ITIMER_REAL, &nval, &oval);
}

Timer::~Timer() {}

void signalhandler(int sig)
{
    cout << __func__ << endl;
    exit(0);
}