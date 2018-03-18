#ifndef _TS_CONTAINER_H_
#define _TS_CONTAINER_H_

#include "task.h"

#include <mutex>

class TSContainer
{
   public:
        TSContainer();
        ~TSContainer();

        void push(Task newTask);
        Task pop();

    private:
        int        full_;
        Task       container_;
        std::mutex m_;
};

#endif /* _TS_CONTAINER_H_ */