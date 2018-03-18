#include "manager.h"

#include <iostream>


Manager::Manager(double intensity, int lifeCycle, std::vector<std::pair<int, double>> *statistic0)
{
    this->intensity_  = intensity;
    this->lifeCycle_  = lifeCycle;
    this->statistic0_ = statistic0;
}

Manager::~Manager() {}

void Manager::doWork(TSQueue &queue)
{
    double globalTime = 0.0;

#ifdef DEBUG
    std::cout << "Test Manager: work start!" << std::endl;
#endif /* DEBUG */

    for (auto i = 0; i < this->lifeCycle_; ++i)
    {

#ifdef DEBUG
        std::cout << "Test Manager added: " << i << " task" << std::endl;
#endif /* DEBUG */

        globalTime += this->calcTow();
        this->statistic0_->push_back(std::pair<int, double>(i, globalTime));

        this->addTask(queue, i, globalTime);
    }

#ifdef DEBUG
    std::cout << "Test Manager: work finish!" << std::endl;
#endif /* DEBUG */

    return;
}

void Manager::addTask(TSQueue &queue, int i, double time)
{
    Task newTask(i, time);
    queue.push_queue(newTask);
}

double Manager::calcTow()
{
    // Obtain a time-based seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // Initialize random generator
    std::default_random_engine generator(seed);

    // Initialize space
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    return (-(std::log(distribution(generator)) / intensity_));
}
