#ifndef _TIMER_H
#define _TIMER_H

class Timer
{
    public:
        Timer(int time);
        ~Timer();
};

void signalhandler(int sig);

#endif /* _TIMER_H */
