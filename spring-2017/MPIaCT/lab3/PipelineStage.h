#ifndef _PIPELINE_STAGE_
#define _PIPELINE_STAGE_

#include "Element.h"

#include <vector>

using namespace std;

class Stage
{
    private:
        std::vector<Stage> *pipeline_;

    public:
        int stageNum_;
        class Element *cpu_;
        class Element *queue_;

        Stage(int stageNum, std::vector<Stage> *pipeline);
        ~Stage();

        int runStageAsync();
        int runStageSync();

        void showInfo();
};

#endif /* _PIPELINE_STAGE_ */
