#include "Element.h"
#include "main.h"

Element::Element(int stages, int type)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(6, 10);
    tickCommon_ = 0;

    if (type) {
        for (auto i = 0; i < stages; ++i) {
            tickStage_.push_back(dis(gen));
            //std::cout << tickStage_[i] << std::endl;
        }
    } else {
        for (auto i = 0; i < stages; ++i) {
            tickStage_.push_back(10);
        }
    }
    //std::cout << std::endl;
}

Element::~Element() {}

int Element::runElement(int stageNum)
{
    tickStage_[stageNum] -= 1;
    tickCommon_++;
/*
   std::cout << "Element at stage : " << stageNum << std::endl;
    std::cout << "Time-to-life left: " << tickStage_[stageNum] << std::endl;
*/
    if (tickStage_[stageNum] > 0) {
        //std::cout << "ELEMENT_STAGE_NOT_COMPLETE" << std::endl;
        return ELEMENT_STAGE_NOT_COMPLETE;
    } else {
        //std::cout << "ELEMENT_STAGE_COMPLETE" << std::endl;
        return ELEMENT_STAGE_COMPLETE;
    }
}

const int Element::getLifeTime()
{
     return tickCommon_;
}