#ifndef _CLASSES_
#define _CLASSES_

#include <iostream>
#include <vector>

#define TICKS 10

enum {
    TICK_COMPLETE   = 0,
    TICK_UNCOMPLETE = 1,
    TICK_END        = 2
};

enum {
    VOWEL_CHAR     = 0,
    CONSONANT_CHAR = 1
};

enum {
    READER_1 = 0,
    READER_2 = 1,
    READER_3 = 2,
    WRITER_1 = 3
};

using namespace std;

class Process
{
    private:
        int               id_;
        int               tick_;
        std::vector<char> *globalArray_;
        std::vector<char> localArray_;

        int local_tick_;

    public:
        Process(int id, int tick, vector<char> *array);
        ~Process();
        int checkChar(char ch);
        int runProcess();
        void printStat();
        void flushOut();
};

#endif /* _CLASSES_ */
