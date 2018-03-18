#ifndef _WORKER_H_
#define _WORKER_H_

#include "thread-safe-queue.h"
#include "thread-safe-container.h"

#include <vector>
#include <algorithm>

class Worker
{
    public:
        Worker();
        Worker(int                                   cycle, 
               int                                   num,
               double                                intensity,
               TSQueue                              *queue,
               TSContainer                          *container,
               std::vector <std::pair<int, double>> *statistic1,
               std::vector <std::pair<int, double>> *statistic2);

        ~Worker();

        void doWork();
        double calcSigma();

    private:
        int                                   lifeCycle_;
        int                                   number_;
        double                                intensity_;
        TSQueue                              *queue_;
        TSContainer                          *container_;
        std::vector <std::pair<int, double>> *statistic1_;
        std::vector <std::pair<int, double>> *statistic2_;
};

#endif /* _WORKER_H_ */