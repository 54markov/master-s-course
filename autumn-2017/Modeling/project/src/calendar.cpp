#include "calendar.h"

#include <queue>
#include <string>
#include <chrono>
#include <random>
#include <iostream>

static double currentTime = 0.0;

static struct event *calendar = NULL;

static std::queue<Task> queue1;
static std::queue<Task> queue2;

/*****************************************************************************/
/* Generating random exponentional time                                      */
/* @input : lambda                                                           */
/* @output: time                                                             */
/*****************************************************************************/
double getRandExp(double lambda)
{
    // Obtain a time-based seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // Initialize random generator
    std::default_random_engine generator(seed);

    // Initialize space
    //std::uniform_real_distribution<double> distribution(0.0, 1.0);

    std::exponential_distribution<double> distribution(lambda);

    return distribution(generator);

    //return -(std::log(distribution(generator)) / lambda);
}

/*****************************************************************************/
/* Push to calendar new event                                                */
/* @input: new event                                                         */
/*****************************************************************************/
void calendarPush(struct event *newEvent)
{
    auto *event = calendar;

    if (!calendar)
    {
        newEvent->next = NULL;
        newEvent->prev = NULL;
        calendar = newEvent;
        return;
    }

    while (event)
    {
        if (event->time > newEvent->time)
        {
            newEvent->next = event;
            newEvent->prev = event->prev;

            if (event->prev)
            {
                event->prev->next = newEvent;
            }
            else
            {
                calendar = newEvent;
            }
            event->prev = newEvent;

            return;
        }

        if (!event->next)
        {
            newEvent->next = NULL;
            newEvent->prev = event;
            event->next    = newEvent;
            return;
        }

        event = event->next;
    }
}

/*****************************************************************************/
/* Pop from calendar new event                                               */
/* @output: event                                                            */
/*****************************************************************************/
struct event *calendarPop()
{
    struct event *event = calendar;
    event->next->prev   = NULL;
    calendar            = event->next;

    return event;
}

/*****************************************************************************/
/* Shedule and create new event to calendar                                  */
/* @input: function pointer                                                  */
/*         schedule time                                                     */
/*****************************************************************************/
void schedule(void (*f)(Statistic &s, int t, double l), double time)
{
    auto *newEvent = new struct event;

    if (!newEvent)
    {
        throw std::string("Memory allocation error!"); // Throw exeption
    }

    if (f == E1)
    {
#ifdef DEBUG
        std::cout << "[" << __FUNCTION__ << "]"
                  << " planned E1 at : "
                  << time << std::endl;
#endif /* DEBUG */
    }
    else if (f == E2)
    {
#ifdef DEBUG
        std::cout << "[" << __FUNCTION__ << "]"
                  << " planned E2 at : "
                  << time << std::endl;
#endif /* DEBUG */
    }
    else if (f == E3)
    {
#ifdef DEBUG
        std::cout << "[" << __FUNCTION__ << "]"
                  << " planned E3 at : "
                  << time << std::endl;
#endif /* DEBUG */
    }
    else if (f == E4)
    {
#ifdef DEBUG
        std::cout << "[" << __FUNCTION__ << "]"
                  << " planned E4 at : "
                  << time << std::endl;
#endif /* DEBUG */
    }
    else
    {
        delete newEvent;
        throw std::string("Event error!"); // Throw exeption
    }

    newEvent->time = time;
    newEvent->proc = f;

    calendarPush(newEvent);
}

/*****************************************************************************/
/* Main simulate function                                                    */
/* @input: statistic object                                                  */
/*         test number                                                       */
/*         modeling time                                                     */
/*         vector lambda                                                     */
/*****************************************************************************/
int simulate(Statistic          &statistic,
             int                 testNum,
             double              modelTime,
             std::vector<double> lambda)
{
    auto periodicTime = 0.0;

#ifdef DEBUG
    std::cout << "[" << __FUNCTION__ << "]"
              << " Monitor: starting simultation"
              << std::endl;
#endif /* DEBUG */

    while (calendar != NULL && currentTime <= modelTime)
    { 
        auto currentEvent = calendarPop();

#ifdef DEBUG
        std::cout << "[" << __FUNCTION__ << "]"
                  << " Monitor: current time is "
                  << currentTime
                  << " running event" << std::endl;
#endif /* DEBUG */

        // Changing time to current event time
        currentTime = currentEvent->time;

        if (currentEvent->proc == E1)
        {
            currentEvent->proc(statistic, testNum, lambda[0]);
        }
        else if (currentEvent->proc == E2)
        {
            currentEvent->proc(statistic, testNum, lambda[1]);
        }
        else if (currentEvent->proc == E3)
        {
            currentEvent->proc(statistic, testNum, lambda[2]);
        }
        else if (currentEvent->proc == E4)
        {
            currentEvent->proc(statistic, testNum, lambda[3]);
        }

        delete currentEvent;

        periodicTime += currentTime;

        if (periodicTime > 1.0)
        {
            periodicTime = 0.0;
            statistic.computeLengths(queue1, queue2);
        }
    }

    if (!calendar)
    {
        std::cout << "[" << __FUNCTION__ << "]"
                  << " Monitor: ending simultation (calendar is empty)"
                  << std::endl;

        return CALENDAR_EMPTY;
    }

    std::cout << "[" << __FUNCTION__ << "]"
              << " Monitor: ending simultation (modeling time expired)"
              << std::endl;

    return TIME_EXPIRED;
}

/*****************************************************************************/
/* Generating income stream to queue 1                                       */
/* @input: statistic object                                                  */
/*         test number                                                       */
/*         lambda                                                            */
/*****************************************************************************/
void E1(Statistic &statistic, int testNum, double lambda)
{
#ifdef DEBUG
    std::cout << "[" << __FUNCTION__ << "]"
              << " Event: current time is : "
              << currentTime << std::endl;
#endif /* DEBUG */

    queue1.push(Task(currentTime, -1.0, true));

    schedule(E1, currentTime + getRandExp(lambda));
}

/*****************************************************************************/
/* Generating income stream to queue 2                                       */
/* @input: statistic object                                                  */
/*         test number                                                       */
/*         lambda                                                            */
/*****************************************************************************/
void E2(Statistic &statistic, int testNum, double lambda)
{
#ifdef DEBUG
    std::cout << "[" << __FUNCTION__ << "]"
              << " Event: current time is : "
              << currentTime << std::endl;
#endif /* DEBUG */

    queue2.push(Task(currentTime, -1.0, true));

    schedule(E2, currentTime + getRandExp(lambda));
}

/*****************************************************************************/
/* Generating worker 1 life cycle                                            */
/* @input: statistic object                                                  */
/*         test number                                                       */
/*         lambda                                                            */
/*****************************************************************************/
void E3(Statistic &statistic, int testNum, double lambda)
{
    static Task localTask;
    static bool isFree = true;

#ifdef DEBUG
    std::cout << "[" << __FUNCTION__ << "]"
              << " Event: current time is : "
              << currentTime << std::endl;
#endif /* DEBUG */

    if (isFree)
    {
        if (!queue1.empty())
        {
            localTask = queue1.front();
            queue1.pop();
            isFree = false;
        }
    }
    else
    {
        if (localTask.first)
        {
            localTask.first = false;
            queue2.push(localTask);
        }
        else
        {
            localTask.end = currentTime;
            statistic.pushTask(localTask);
        }

        isFree = true;
    }

    if (testNum == 1)
    {
        schedule(E3, currentTime + getRandExp(lambda));
    }
    else if (testNum == 2)
    {
        lambda = 1.0 / lambda;
        schedule(E3, currentTime + lambda);
    }
}

/*****************************************************************************/
/* Generating worker 2 life cycle                                            */
/* @input: statistic object                                                  */
/*         test number                                                       */
/*         lambda                                                            */
/*****************************************************************************/
void E4(Statistic &statistic, int testNum, double lambda)
{
    static Task localTask;
    static bool isFree = true;

#ifdef DEBUG
    std::cout << "[" << __FUNCTION__ << "]"
              << " Event: current time is : "
              << currentTime << std::endl;
#endif /* DEBUG */

    if (isFree)
    {
        if (!queue2.empty())
        {
            localTask = queue1.front();
            queue2.pop();
            isFree = false;
        }
    }
    else
    {
        if (localTask.first)
        {
            localTask.first = false;
            queue1.push(localTask);
        }
        else
        {
            localTask.end = currentTime;
            statistic.pushTask(localTask);
        }

        isFree = true;
    }

    if (testNum == 1)
    {
        schedule(E4, currentTime + getRandExp(lambda));
    }
    else if (testNum == 2)
    {
        lambda = 1.0 / lambda;
        schedule(E4, currentTime + lambda);
    }
}

void resetModelingTime()
{
    currentTime = 0.0;
}

void resetQueues()
{
    while(!queue1.empty())
    {
        queue1.pop();
    }

    while(!queue2.empty())
    {
        queue2.pop();
    }
}
