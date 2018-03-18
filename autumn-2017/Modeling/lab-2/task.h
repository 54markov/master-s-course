#ifndef _TASK_H_
#define _TASK_H_

class Task
{
    public:
        Task();
        Task(int number, double time);
        ~Task();

        int    getNumber();
        double getQueueTime();
        double getWorker1Time();
        double getWorker2Time();

        void setNumber(int number);
        void setQueueTime(double time);
        void setWorker1Time(double time);
        void setWorker2Time(double time);

        void addToQueueTime(double time);

    private:
        int    number_;      // Task number
        double queueTime_;   // Queue Life time
        double worker1Time_; // Worker 1 life time 
        double worker2Time_; // Worker 2 life time 
};

#endif /* _TASK_H_ */