#include "functions.h"

void runSimulate(int &allR,
                 int &allW,
                 int &completeR,
                 int &completeW)
{
    std::vector<Process*> queueR;
    std::vector<Process*> queueW;

    Process  *runProc;
    int cpuIsBusy = 0;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (auto i = 0; i < 100000; ++i) {
        auto p = 0.0;

        p = dis(gen);

        if ((p < 0.05) && (p > 0.005)) {
            Process *p = new Process("reader");
            queueR.push_back(p);
            allR++;
        } else if (p < 0.005) {
            Process *p = new Process("writer");
            queueW.push_back(p);
            allW++;
        }

        if (cpuIsBusy) {
            if (runProc->runProc() == TICK_COMPLETE) {
                cpuIsBusy = 0;
                if (runProc->getProcId().compare(std::string("reader")) == 0) {
                    completeR++;
                } else {
                    completeW++;
                }

                delete(runProc);
            } else {
                continue;
            }
        } else {
            if (queueR.size() != 0) {
                cpuIsBusy = 1;
                runProc   = queueR.back();
                runProc->runProc();
                queueR.pop_back();
            } else if (queueW.size() != 0) {
                cpuIsBusy = 1;
                runProc   = queueW.back();
                runProc->runProc();
                queueW.pop_back();
            }
        }
    }


    queueR.clear();
    queueW.clear();
}

void showStat(int &allR,
              int &allW,
              int &completeR,
              int &completeW)
{
    std::cout << "Whole numbers of read           processes " << allR << std::endl;
    std::cout << "Whole numbers of write          processes " << allW << std::endl;
    std::cout << "Whole numbers of complete read  processes " << completeR << std::endl;
    std::cout << "Whole numbers of complete write processes " << completeW << std::endl;
}
