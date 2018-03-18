#ifndef _POINT_
#define _POINT_

#include <vector>

struct Point
{
    double x1;    // upper ellipsiod intersection point
    double y1;    // upper ellipsiod intersection point
    double z1;    // upper ellipsiod intersection point

    double x2;    // lower ellipsiod intersection point
    double y2;    // lower ellipsiod intersection point
    double z2;    // lower ellipsiod intersection point

    double x0;    //draw by this point
    double y0;    //draw by this point
    double z0;    //draw by this point

    double power; // X-ray power of th the points
    int    type;  // type of point
};


int getEllipsoidIntersectionPoints(double x, double y);

double getEllipsoidIntersectionPointsZ(double x, double y);

double getEllipsoidIntersectionPointsY(double x, double z);

int showRayGridPower(const std::vector<std::vector<struct Point>> &grid);

double getAngleRatio(double x1, double z1, double x2, double z2);

void getSecondPoints(double &x2, 
                     double &z2, 
                     double &power,
                     double &x0,
                     double &y0,
                     double &z0,
                     double x1,
                     double z1);

void convertGrid(const std::vector<std::vector<struct Point>> &a,
	             std::vector<std::vector<struct Point>> &b);

#endif /* _POINT_ */