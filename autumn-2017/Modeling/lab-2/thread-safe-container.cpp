#include "thread-safe-container.h"

TSContainer::TSContainer() { this->full_ = 0; }
TSContainer::~TSContainer() {}

void TSContainer::push(Task newTask)
{
    while (1)
    {
        this->m_.lock();

        if (!this->full_)
        {
            this->container_ = newTask;
            this->full_      = 1;
            this->m_.unlock();
            break;
        }

        this->m_.unlock();
    }
}

Task TSContainer::pop()
{
    Task copy;
    
    while (1)
    {
        this->m_.lock();

        if (this->full_)
        {
            copy = this->container_;
            this->full_ = 0;
            this->m_.unlock();
            break;
        }

        this->m_.unlock(); 
    }

    return copy;
}
