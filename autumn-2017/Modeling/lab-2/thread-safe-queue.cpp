#include "thread-safe-queue.h"

TSQueue::TSQueue() {}
TSQueue::~TSQueue() {}

void TSQueue::push_queue(Task newTask)
{
    this->m_.lock(); 
    this->q_.push(newTask);
    this->m_.unlock();
}

Task TSQueue::pop_queue()
{
    Task copy;
    
    while (1)
    {
        this->m_.lock();
        
        if (this->q_.empty())
        {
            this->m_.unlock();
            continue;
        }
        else
        {
            copy = this->q_.front();
            this->q_.pop();
            this->m_.unlock();
            break;
        }
    }

    return copy;
}

bool TSQueue::is_empty()
{
    if (this->q_.empty())
        return true;

    return false;
}
