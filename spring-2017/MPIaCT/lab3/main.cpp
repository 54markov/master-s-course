#include "main.h"
#include "PipelineStage.h"

#include <vector>

class Element *queueNext[N];

int main(int argc, char const *argv[])
{
    std::vector<Stage> *asyncPipeline = new std::vector<Stage>;
   
    for (auto i = 0; i < N; ++i) {
        class Stage newStage(i, asyncPipeline);
        asyncPipeline->push_back(newStage);
    }
    auto elapsedTicks = 0;
    for (auto count = 0; count < 10000; elapsedTicks++) {
        for (auto i = 0; i < N - 1; ++i) {
            if (asyncPipeline->at(i).runStageAsync() > 0) {
                //std::cout << "define n: " << std::endl;
                //count += 1;
            }
        }
        if (asyncPipeline->at(N-1).runStageAsync() > 0) {
            //std::cout << "define n: " << std::endl;
            count += 1;
        }
    }
    std::cout << "define n: " << N << std::endl;
    std::cout << "asyncPipeline run elapsed ticks for 10000 (elements): " << elapsedTicks << std::endl;

    /*************************************************************************/
/*
    std::vector<Stage> *syncPipeline = new std::vector<Stage>;
    for (auto i = 0; i < N; ++i) {
        class Stage newStage(i, syncPipeline);
        syncPipeline->push_back(newStage);
    }

    elapsedTicks = 0;
    for (auto count = 0; count < 10000; elapsedTicks++) {
        for (auto i = 0; i < N; ++i) {
            if (syncPipeline->at(i).runStageSync() > 0) {
                count += 1;
            }
        }        
    }
    std::cout << "syncPipeline run elapsed ticks for 10000 (elements) : " << elapsedTicks << std::endl;
*/
    return 0;
}
