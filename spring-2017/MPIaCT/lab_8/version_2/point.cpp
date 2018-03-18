#include "point.h"
#include "main.h"
#include "plot.h"

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

/*****************************************************************************/
/*                                                                           */
/* INPUT  : x, y                                                             */
/* RETURN : angle                                                            */
/*                                                                           */
/*****************************************************************************/
static double getAngleRatio(double x2, double y2, double x1, double y1)
{
    return ((y2 - y1) / (x2 - x1));
}

/*****************************************************************************/
/*                                                                           */
/* INPUT  : x, y, z                                                          */
/* RETURN : line lneght                                                      */
/*                                                                           */
/*****************************************************************************/
static double getLineLenght(double x1, double y1, double z1, 
                            double x2, double y2, double z2)
{
    return (sqrtf(((z2 - z1) * (z2 - z1) +
                      (x2 - x1) * (x2 - x1) +
                      (y2 - y1) * (y2 - y1) )));
}
/*****************************************************************************/
/*                                                                           */
/* INPUT  : x, y                                                             */
/* RETURN : power                                                            */
/*                                                                           */
/*****************************************************************************/
double getSurfacePoints(double xSurface, double ySurface)
{
    double zSurface = -5.0;

    /*
     * X-ray source, start point
     */
    //double x1 = 0.0;
    //double y1 = 0.0;
    double z1 = 5.0;

    /* 
     * Line parametric equation
     */
    double l = xSurface;
    double m = ySurface;
    double n = zSurface;

    double t1 = -1.0; // First root
    double t2 = -1.0; // Second root

    /*
     * MATH part
     */
    double a = (144 * (l * l)) + (100 * (m * m)) + (225 * (n * n));
    double b = -(86.4 * l + 40 * m  - 1935 * n);
    double c = 4141.21;
    double D = sqrtf((b * b) - (4 * a * c));

    if (D < 0.0) {
        // Case: no roots
        return 0.0;
    } else if (D > 0.0) {

        t1 = (-b - D) / 2 * a;
        t2 = (-b + D) / 2 * a;


        // Case: two roots
        double Ax = l * t1;
        double Ay = m * t1;
        double Az = n * t1 + z1; // z1 = 5.0

        double Bx = l * t2;
        double By = m * t2;
        double Bz = n * t2 + z1; // z1 = 5.0

        double len = getLineLenght(Ax, Ay, Az, Bx, By, Bz);

        len = len / 100000000;

        double lenInternal = getInternalSurfacePoints(xSurface, ySurface);
        //cout << lenInternal << endl;
        return pow(EXP, -(len + lenInternal));

        //return pow(EXP, -len);
    }
    return 0.0;
}


double getInternalSurfacePoints(double xSurface, double ySurface)
{
    double zSurface = -5.0;

    /*
     * X-ray source, start point
     */
    //double x1 = 0.0;
    //double y1 = 0.0;
    double z1 = 5.0;

    /* 
     * Line parametric equation
     */
    double l = xSurface;
    double m = ySurface;
    double n = zSurface;

    double t1 = -1.0; // First root
    double t2 = -1.0; // Second root

    /*
     * MATH part
     */
    double a = (144 * (l * l)) + (100 * (m * m)) + (225 * (n * n));
    double b = -(86.4 * l + 40 * m  - 1935 * n);
    double c = 4176.85;
    double D = sqrtf((b * b) - (4 * a * c));

    if (D < 0.0) {
        // Case: no roots
        return 0.0;
    } else if (D > 0.0) {

        t1 = (-b - D) / 2 * a;
        t2 = (-b + D) / 2 * a;


        // Case: two roots
        double Ax = l * t1;
        double Ay = m * t1;
        double Az = n * t1 + z1; // z1 = 5.0

        double Bx = l * t2;
        double By = m * t2;
        double Bz = n * t2 + z1; // z1 = 5.0

        double len = getLineLenght(Ax, Ay, Az, Bx, By, Bz);

        len = len / 100000000;

        return len;
    }
    return 0.0;
}