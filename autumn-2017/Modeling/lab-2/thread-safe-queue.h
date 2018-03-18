#ifndef _TS_QUEUE_H_
#define _TS_QUEUE_H_

#include "task.h"

#include <mutex>
#include <queue>

class TSQueue
{
    public:
        TSQueue();
        ~TSQueue();

        void push_queue(Task newTask);
        Task pop_queue();
        bool is_empty();

    private:
        std::queue<Task> q_;
        std::mutex       m_;
};

#endif /* _TS_QUEUE_H_ */