#ifndef _STATISTIC_H_
#define _STATISTIC_H_

#include <queue>
#include <vector>

#include "task.h"

class Statistic
{
    public:
        Statistic(int test, double q1, double q2, double w1, double w2);
        ~Statistic();

        void pushTask(Task t);
        void computeLengths(const std::queue<Task> &q1, const std::queue<Task> &q2);
        void showStatistic();
        double lastTime();

    private:
        std::vector<std::size_t> averageQueueLengths_;
        std::vector<Task>   completedTasks_;

        int test_;
        double q1_, q2_, w1_, w2_;
};

#endif /* _STATISTIC_H_ */
