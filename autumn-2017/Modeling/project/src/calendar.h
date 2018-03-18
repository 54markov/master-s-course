#ifndef _CALENDAR_H_
#define _CALENDAR_H_

#include "statistic.h"

enum
{
    CALENDAR_EMPTY = 1,
    TIME_EXPIRED   = 2
};

struct event 
{    
    double time;                        // Income time
    void (*proc)(Statistic &statistic,
                 int t, double l);      // Function pointer
    struct event *next;                 // Next event
    struct event *prev;                 // Prev event
};

/*****************************************************************************/
/* Generating random exponentional time                                      */
/* @input : lambda                                                           */
/* @output: time                                                             */
/*****************************************************************************/
double getRandExp(double lambda);

/*****************************************************************************/
/* Push to calendar new event                                                */
/* @input: new event                                                         */
/*****************************************************************************/
void calendarPush(struct event *newEvent);

/*****************************************************************************/
/* Pop from calendar new event                                               */
/* @output: event                                                            */
/*****************************************************************************/
struct event *calendarPop();

/*****************************************************************************/
/* Shedule and create new event to calendar                                  */
/* @input: function pointer                                                  */
/*         schedule time                                                     */
/*****************************************************************************/
void schedule(void (*f)(Statistic &s, int t, double l), double time);

/*****************************************************************************/
/* Main simulate function                                                    */
/* @input: statistic object                                                  */
/*         test number                                                       */
/*         modeling time                                                     */
/*****************************************************************************/
int simulate(Statistic          &statistic,
             int                 testNum,
             double              modelTime,
             std::vector<double> lambda);

/*****************************************************************************/
/* Generating income stream to queue 1                                       */
/* @input: statistic object                                                  */
/*         test number                                                       */
/*         lambda                                                            */
/*****************************************************************************/
void E1(Statistic &s, int t, double l);

/*****************************************************************************/
/* Generating income stream to queue 2                                       */
/* @input: statistic object                                                  */
/*         test number                                                       */
/*         lambda                                                            */
/*****************************************************************************/
void E2(Statistic &s, int t, double l);

/*****************************************************************************/
/* Generating worker 1 life cycle                                            */
/* @input: statistic object                                                  */
/*         test number                                                       */
/*         lambda                                                            */
/*****************************************************************************/
void E3(Statistic &s, int t, double l);

/*****************************************************************************/
/* Generating worker 2 life cycle                                            */
/* @input: statistic object                                                  */
/*         test number                                                       */
/*         lambda                                                            */
/*****************************************************************************/
void E4(Statistic &s, int t, double l);

void resetModelingTime();
void resetQueues();

#endif /* _CALENDAR_H_ */
