#include <algorithm>
#include <iostream>

#include "color.h"
#include "statistic.h"


Statistic::Statistic(int test, double q1, double q2, double w1, double w2) :
                     test_(test), q1_(q1), q2_(q2), w1_(w1), w2_(w2) { }

Statistic::~Statistic() { }

void Statistic::pushTask(Task t)
{
    this->completedTasks_.push_back(t);
}

void Statistic::computeLengths(const std::queue<Task> &q1, const std::queue<Task> &q2)
{

#ifdef DEBUG
    std::cout << __FUNCTION__ << ":" << q1.size() + q2.size() << std::endl;
#endif /* DEBUG */

    this->averageQueueLengths_.push_back(q1.size() + q2.size());
}

double Statistic::lastTime()
{
    if (!this->completedTasks_.empty())
    {
        auto lastTask = *(this->completedTasks_.end() - 1);
        return lastTask.end;
    }
    return 0.0;
}

void Statistic::showStatistic()
{
    auto qSize = this->averageQueueLengths_.size();
    auto tSize = completedTasks_.size();
    auto qSum  = 0;
    auto tSum  = 0.0;
    auto avgSize = 0.0;
    auto avgTime = 0.0;

    std::for_each(this->averageQueueLengths_.begin(),
                  this->averageQueueLengths_.end(), [&](int len)
    {
        qSum += len;
    });

    if (qSum == 0 || qSize == 0)
    {
        avgSize = 0.0;
    }
    else
    {
        avgSize = static_cast<double>(qSum) / static_cast<double>(qSize);
    }

    std::for_each(this->completedTasks_.begin(),
                  this->completedTasks_.end(), [&](Task t)
    {
#ifdef DEBUG
        std::cout << t.end << " - " << t.start <<  " = " << t.end - t.start << std::endl;
#endif /* DEBUG */
        tSum += (t.end - t.start);
    });

    if (tSum == 0.0 || tSize == 0)
    {
        avgTime = 0.0;
    }
    else
    {
        avgTime = static_cast<double>(tSum) / static_cast<double>(tSize);
    }

    if (this->test_ == 2)
    {
        this->w1_ = 1.0 / this->w1_;
        this->w2_ = 1.0 / this->w2_;
    }

    Color::Modifier blue(Color::FG_BLUE);
    Color::Modifier def(Color::FG_DEFAULT);

    std::cout << "Show statistic:          " << std::endl
              << "Queue-1  λ(" << this->q1_ << ")" << std::endl
              << "Queue-2  λ(" << this->q2_ << ")" << std::endl
              << "Worker-1 λ(" << this->w1_ << ")" << std::endl
              << "Worker-2 λ(" << this->w2_ << ")" << std::endl << std::endl
              << blue << "Average queue lengths  : " << avgSize
              << " (tasks)" << std::endl
              << "Average life-time task : " << avgTime 
              << " (model time)" << def << std::endl;
}
