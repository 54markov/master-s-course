#include <iostream>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <vector>

#include <algorithm>
#include <chrono>
#include <random>

using namespace std;

const int GLOBAL_CYCLE = 100;

/*****************************************************************************/
/*                                                                           */      
/*      +-------------+                                                      */
/*     \|/            |                                                      */
/* --->[Q]---------->[W1]---------->[W2]---------->                          */
/*          w1=work  /|\   w1=free   |    w2=free                            */
/*                    |              |                                       */
/*                    +--------------+                                       */
/*                         w2=free                                           */
/*****************************************************************************/

std::vector <pair<int, double>> statistic0;
std::vector <pair<int, double>> statistic1;
std::vector <pair<int, double>> statistic2;

class TSQueue // thread safe queue
{
    public:
        TSQueue() {}
        ~TSQueue() {}

        void push_queue(Task newTask)
        {
            mutex_.lock();
            taskQueue_.push(newTask);
            mutex_.unlock();
        }

        Task pop_queue()
        {
            Task copy;
            while(1) {
                mutex_.lock();
                if (taskQueue_.empty()) {
                    mutex_.unlock();
                    continue;
                } else {
                    copy = taskQueue_.front();
                    taskQueue_.pop();
                    mutex_.unlock();
                    break;
                }
            }
            return copy;
        }

        bool is_empty()
        {
            if (taskQueue_.empty()) {
                return true;
            } else {
                return false;
            }
        }

    private:
        queue<Task> taskQueue_;
        mutex mutex_;
};

class ManagerQueue
{
    private:
        int cycle_;
        double intensity_;
    
    public:
        ManagerQueue(double param, int cnt): intensity_(param), cycle_(cnt) {}
        ~ManagerQueue() {}

        void addTask(TSQueue &tsQueue, int i, double time)
        {
            Task newTask(i, time);
            //newTask.number = i;
            //newTask.queue  = time;

            tsQueue.push_queue(newTask);
        }

        double calc_tow()
        {
            // obtain a time-based seed
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            // initialize random generator
            std::default_random_engine generator(seed);
            // initialize space
            std::uniform_real_distribution<double> distributionR(0.0, 1.0);

            auto r = distributionR(generator);

            double x = -(log(r) / (int)intensity_);

            return x;
        }

        void run(TSQueue &tsQueue)
        {
            double globalTime = 0.0;
            std::cout << "Test Manager: work start!" << std::endl;
            for (auto i = 0; i < cycle_; ++i)
            {
                std::cout << "Test Manager added: " << i << " task" << std::endl;
                globalTime += calc_tow();
                statistic0.push_back(pair<int, double> (i, globalTime));
                addTask(tsQueue, i, globalTime);
            }
            std::cout << "Test Manager: work finish!" << std::endl;
            return;
        }
};

class TSContainer
{
    private:
        int full;
        Task container;
        mutex mutex_;
    public:
        TSContainer() { full = 0; }
        ~TSContainer() {}

        void push(Task newTask)
        {
            while (1) {
                mutex_.lock();
                if (full == 0) {
                    container = newTask;
                    full = 1;
                    mutex_.unlock();
                    break;
                }
                mutex_.unlock(); 
            }
        }

        Task pop()
        {
            Task copy_container;
            while (1) {
                mutex_.lock();
                if (full == 1) {
                    copy_container = container;
                    full = 0;
                    mutex_.unlock();
                    break;
                }
                mutex_.unlock(); 
            }
            return copy_container;
        }
};

TSQueue testQueue; // global thread-safe queue
TSContainer testContainer;

class Worker
{
    private:
        int cycle_;
        int number_;
        double intensity_;

    public:
        Worker(int cycle, int num, double param)
        {
            cycle_ = cycle;
            number_ = num;
            intensity_= 1.0 / param;
        }

        ~Worker() {}

        double calc_sigma()
        {
            // obtain a time-based seed
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            // initialize random generator
            std::default_random_engine generator(seed);
            // initialize space
            std::uniform_real_distribution<double> distributionR(0.0, 1.0);

            auto r = distributionR(generator);

            double x = -(log(r) / intensity_);

            return x;
        }

        void test()
        {
            cout << "Test Worker " << number_ << ": work start!" << endl;

            if (number_ == 1)
            {
                for (auto i = 0; i < cycle_; ++i)
                {
                    auto popedTask = testQueue.pop_queue();
                    popedTask.addToQueueTime(calc_sigma());
                    statistic1.push_back(pair<int, double>(popedTask.getNumber(), popedTask.getQueueTime()));
                    testContainer.push(popedTask);                  
                    std::cout << "Test Worker " << number_ << ": task - take!" << std::endl;
                }   
            }
            else
            {
                for (auto i = 0; i < cycle_; ++i)
                {
                    auto popedTask = testContainer.pop();
                    popedTask.addToQueueTime(calc_sigma());
                    statistic2.push_back(pair<int, double>(popedTask.getNumber(), popedTask.getQueueTime()));  
                    std::cout << "Test Worker " << number_ << ": task - take!" << std::endl;
                }
            }
            std::cout << "Test Worker " << number_ << ": work finish!" << std::endl;
        }   
};

void runWorker(int num, double param)
{
    Worker testWorker(GLOBAL_CYCLE, num, param);
    testWorker.doWork();
}

void run_test(int i)
{
    if (i == 0) {
        cout << "Run test worker1 = 1.5, worker2 = 3.0 " << endl;

        ManagerQueue testManager(1.0, GLOBAL_CYCLE); // task queue manager

        thread thWorker1(runWorker, 1, 1.5, testQueue, testContainer);
        thread thWorker2(runWorker, 2, 3.0, testQueue, testContainer);

        testManager.run(testQueue);

        thWorker1.join();
        thWorker2.join();

        cout << endl;
        cout << statistic0.size() << endl;
        cout << statistic1.size() << endl;
        cout << statistic2.size() << endl;
        cout << endl;

        std::cout << "Queue\t    Worker1\t    Worker2\t    ElapsedTime" << std::endl;

        for (auto i = 0; i < GLOBAL_CYCLE; ++i)
        {
            cout << statistic0[i].second << "\t    ";
            cout << statistic1[i].second << "\t    ";
            cout << statistic2[i].second << "\t    ";
            cout << statistic2[i].second - statistic0[i].second << endl;
        }

        if (testQueue.is_empty())
        {
            std::cout << "Global queue - empty!" << std::endl;
        } 
        else
        {
            std::cout << "Global queue - non empty!" << std::endl;
        }
    } 
    else
    {
        ManagerQueue testManager(1.0, GLOBAL_CYCLE); // taes queue manager

        thread thWorker1(runWorker, 1, 3.0);
        thread thWorker2(runWorker, 2, 1.5);

        testManager.run(testQueue);

        thWorker1.join();
        thWorker2.join();

        cout << endl;
        cout << statistic0.size() << endl;
        cout << statistic1.size() << endl;
        cout << statistic2.size() << endl;
        cout << endl;

        std::cout << "Queue\t    Worker1\t    Worker2\t    ElapsedTime" << std::endl;

        for (auto i = 0; i < GLOBAL_CYCLE; ++i)
        {
            cout << statistic0[i].second << "\t    ";
            cout << statistic1[i].second << "\t    ";
            cout << statistic2[i].second << "\t    ";
            cout << statistic2[i].second - statistic0[i].second << endl;
        }

        if (testQueue.is_empty())
        {
            std::cout << "Global queue - empty!" << std::endl;
        }
        else
        {
            std::cout << "Global queue - non empty!" << std::endl;
        }
    }
}

int main(int argc, char const *argv[])
{
    try
    {
        if (argc < 2)
            throw std::string("Not enought arguments");

        std::string option(argv[1]);

        if (!option.compare("a"))
            run_test(0);
        else if (!option.compare("b"))
            run_test(1);
        else
            throw std::string("Not recognized option");
    }
    catch (std::string err)
    {
        std::cerr << err << std::endl;
        return -1;
    }

    return 0;
}
