#include "worker.h"

#include <iostream>
#include <string>
#include <queue>
#include <thread>
#include <mutex>

#include <chrono>
#include <random>

Worker::Worker() {}

Worker::Worker(int                                   cycle, 
               int                                   num,
               double                                intensity,
               TSQueue                              *queue,
               TSContainer                          *container,
               std::vector <std::pair<int, double>> *statistic1,
               std::vector <std::pair<int, double>> *statistic2)
{
    this->lifeCycle_  = cycle;
    this->number_     = num;
    this->intensity_  = (1.0 / intensity);
    this->queue_      = queue;
    this->container_  = container;
    this->statistic1_ = statistic1;
    this->statistic2_ = statistic2;
}

Worker::~Worker() {}

void Worker::doWork()
{

#ifdef DEBUG
    std::cout << "Test Worker " << this->number_ << ": work start!" << std::endl;
#endif /* DEBUG */

    for (auto i = 0; i < this->lifeCycle_; ++i)
    {
        if (this->number_ == 1)
        {
            auto popedTask = this->queue_->pop_queue();

            popedTask.addToQueueTime(this->calcSigma());

            this->statistic1_->push_back(std::pair<int, double>(popedTask.getNumber(), popedTask.getQueueTime()));

            this->container_->push(popedTask);

#ifdef DEBUG
            std::cout << "Test Worker " << this->number_ << ": task - take!" << std::endl;
#endif /* DEBUG */

        }
        else if (this->number_ == 2)
        {
            auto popedTask = this->container_->pop();

            popedTask.addToQueueTime(this->calcSigma());

            this->statistic2_->push_back(std::pair<int, double>(popedTask.getNumber(), popedTask.getQueueTime()));

#ifdef DEBUG
            std::cout << "Test Worker " << this->number_ << ": task - take!" << std::endl;
#endif /* DEBUG */

        }
    }

#ifdef DEBUG
    std::cout << "Test Worker " << this->number_ << ": work finish!" << std::endl;
#endif /* DEBUG */

}

double Worker::calcSigma()
{
    // Obtain a time-based seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // Iitialize random generator
    std::default_random_engine generator(seed);

    // Initialize space
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    return (-(std::log(distribution(generator)) / this->intensity_));
}
