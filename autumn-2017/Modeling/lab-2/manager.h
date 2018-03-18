#ifndef _MANAGER_H_
#define _MANAGER_H_

#include "thread-safe-queue.h"

#include <vector>
#include <algorithm>

class Manager
{
    public:
        Manager(double intensity, int lifeCycle,
            std::vector<std::pair<int, double>> *statistic0);

        ~Manager();

        void doWork(TSQueue &queue);
        void addTask(TSQueue &queue, int i, double time);
        double calcTow();

    private:
        int    lifeCycle_;
        double intensity_;

        std::vector<std::pair<int, double>> *statistic0_;
};

#endif /* _MANAGER_H_ */
