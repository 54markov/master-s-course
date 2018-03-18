#ifndef _ELEMENT_
#define _ELEMENT_

#include <iostream>
#include <vector>
#include <random>

class Element
{
    private:
        int              tickCommon_;
        std::vector<int> tickStage_;

    public:
        Element(int stages, int type);
        ~Element();

        int       runElement(int stageNum);
        const int getLifeTime();
};

#endif /* _ELEMENT_ */
