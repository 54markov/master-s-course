#include "task.h"

Task::Task() {}

Task::Task(int number, double time)
{
    this->number_      = number;
    this->queueTime_   = time;
    this->worker1Time_ = 0.0;
    this->worker2Time_ = 0.0;
}

Task::~Task() {}

int    Task::getNumber()      { return this->number_; }
double Task::getQueueTime()   { return this->queueTime_; }
double Task::getWorker1Time() { return this->worker1Time_; }
double Task::getWorker2Time() { return this->worker2Time_; }

void Task::setNumber(int number)       { this->number_ = number; }
void Task::setQueueTime(double time)   { this->queueTime_ = time; }
void Task::setWorker1Time(double time) { this->worker1Time_ = time; }
void Task::setWorker2Time(double time) { this->worker2Time_ = time; }

void Task::addToQueueTime(double time) { this->queueTime_ += time; }