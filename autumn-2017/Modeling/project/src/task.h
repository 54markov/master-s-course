#ifndef _TASK_H_
#define _TASK_H_

class Task
{
    public:
    	Task(double s = -1.0, double e = -1.0, bool f = false);
        double start;
        double end;
        bool   first;
};

#endif /* _TASK_H_ */
