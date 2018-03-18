#ifndef _POINT_
#define _POINT_

#include <vector>

struct Point
{
/*
    double x1;    // upper ellipsiod intersection point
    double y1;    // upper ellipsiod intersection point
    double z1;    // upper ellipsiod intersection point

    double x2;    // lower ellipsiod intersection point
    double y2;    // lower ellipsiod intersection point
    double z2;    // lower ellipsiod intersection point

    double x0;    //draw by this point
    double y0;    //draw by this point
    double z0;    //draw by this point
*/
    double power; // X-ray power of th the points
/*
    int    type;  // type of point
*/
};

double getSurfacePoints(double xSurface, double ySurface);
double getInternalSurfacePoints(double xSurface, double ySurface);

#endif /* _POINT_ */