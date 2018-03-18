#ifndef _FIELD_
#define _FIELD_

#include <vector>

class Field
{
    private:
        std::vector<std::vector<char> > field_;
        std::vector<std::pair<int, int> > wayPoints_;

    public:
        Field(int x = 100, int y  = 100);
        ~Field();
        
        void printField();
        int setWayPoint(int x, int y);
        void generateWayPoints(int wayPoints);

        void getPathBetweenWayPoints();
        int getPathBetweenWayPoints_(std::pair<int, int> a, std::pair<int, int> b);
        void permutation(std::vector<std::pair<int, int> > vecPair);

};

#endif /* _FIELD_ */