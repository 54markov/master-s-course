#include "point.h"
#include "main.h"
#include "plot.h"

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

double getEllipsoidIntersectionPointsZ(double x, double y)
{
    double sqA = 0.25;         // 0.5
    double sqB = 0.36;         // 0.6
    double sqC = 0.16;         // 0.4
    double z0  = 0.7;          // z0 = 0.7

    x = (x - 0.3) * (x - 0.3); // x0 = 0.3
    y = (y - 0.2) * (y - 0.2); // y0 = 0.2

    double a = x / sqA;
    double b = y / sqB;

    double tmp = ((1.0 - a - b) * sqC);

    if (tmp < 0) {
        tmp = -1.0 * tmp;
    }

    return sqrtf(tmp) + z0; // return z point
}

double getEllipsoidIntersectionPointsY(double x, double z)
{
    double sqA = 0.25;         // 0.5
    double sqB = 0.36;         // 0.6
    double sqC = 0.16;         // 0.4
    double y0  = 0.2;

    x = (x - 0.3) * (x - 0.3); // x0 = 0.3
    z = (z - 0.7) * (z - 0.7); // z0 = 0.7

    double a = x / sqA;
    double c = z / sqC;

    double tmp = ((1.0 - a - c) * sqB);

    if (tmp < 0) {
        tmp = -1.0 * tmp;
    }

    return sqrtf(tmp) + y0; // return y point
}

int getEllipsoidIntersectionPoints(double x, double y)
{
    double sqA = 0.25; // 0.5
    double sqB = 0.36; // 0.6

    x = (x - 0.3) * (x - 0.3); // x0 = 0.3
    y = (y - 0.2) * (y - 0.2); // y0 = 0.2

    double a = x / sqA;
    double b = y / sqB;

    double result = a + b - 1.0;

    if (result == 0.0) {
        return INTERSECTION;
    } else if (result < 0.0) {
        return INSIDE;
    } else {
        return OUTSIDE;
    }
}

int showRayGridPower(const vector<vector<struct Point>> &grid)
{
    for (auto i: grid) {
        for (auto j: i) {
            cout << j.type;
        }

    }
    return 0;
}

double getAngleRatio(double x2, double z2, double x1, double z1)
{
    return ((z2 - z1) / (x2 - x1));
}

static double getLineLeght(double z1, double xy1, double z0, double xy0)
{
    double lenght = sqrtf( ((z1 - z0) * (z1 - z0) + (xy1 - xy0) * (xy1 - xy0)) );

    if (lenght < 0.0) {
        lenght = lenght * -1.0;
    }
    return lenght;
}

static double getSurfacePointXY(double zAxesCoord, double angle, double xyAxsesCoord)
{
    /*
     *    B
     *    |\
     *    | \
     *   a|  \ c
     *    |   \
     *    |____\
     *    C  b  A
     *
     *   b = a * tg(A)
     */

    double a = getLineLeght(zAxesCoord, xyAxsesCoord, 0, xyAxsesCoord);



    double b = a * atan(angle);

#ifdef DEBUG
    cout << __FUNCTION__ << endl;
    cout << "Angle = " << angle << endl;
    cout << "Len A = " << a << endl;
    cout << "Len b = " << b << endl;
    cout << "Coord = " << xyAxsesCoord + b << endl;
    cout << endl;
#endif

    if (((xyAxsesCoord + b) > 2.0) || ((xyAxsesCoord + b) < -2.0)) {
        cerr << __FUNCTION__ << "ERROR: getting coordinates" << endl;
        exit (-1);
    }

    return xyAxsesCoord + b;
}

/*
 * Return: x1 and x2 
 *
 */
void getSecondPoints(double &x2, 
                     double &z2, 
                     double &power,
                     double &x0,
                     double &y0,
                     double &z0,
                     double x1,
                     double z1)
{
    /*
     *    z axes
     *     |
     *     & - ray sourse (0, 5)
     *     | .
     *     |  .
     *     |    & - x1 z1 - input
     *     |     .--------------------------
     *     |      .                        | - l lenght
     *     |       .------------------------
     *     |        ?  - x2 z2 - return
     *     |
     *  --------------------> x axes
     */

    /*
     * From: http://www.ambrsoft.com/TrigoCalc/Circles2/Ellipse/EllipseLine.htm
     */

    double m = getAngleRatio(0.0, 15.0, x1, z1);

    double c = 15.0;

    double h = 0.3;
    double k = 0.7;

    double e = c - k;
    double sigma = c + m * h;

    double a = 0.5, sqA = 0.25;
    double b = 0.4, sqB = 0.16;

    double sqm = m * m;
    double sqsigma = sigma * sigma;

    double tmp = (sqA * sqm) + (sqB - sqsigma) - (k * k) + (2 * sigma * k);

    double rootX1 = ((h * sqB) - (m * sqA * e) + (a * b * sqrtf(tmp))) / ((sqA * (m * m)) + sqB);
    double rootX2 = ((h * sqB) - (m * sqA * e) - (a * b * sqrtf(tmp))) / ((sqA * (m * m)) + sqB);

    double rootZ1 = ((sqB * sigma) + (k * sqA * (m * m)) + (a * b * m * sqrtf(tmp))) / ((sqA * (m * m)) + sqB);
    double rootZ2 = ((sqB * sigma) + (k * sqA * (m * m)) - (a * b * m * sqrtf(tmp))) / ((sqA * (m * m)) + sqB);
/*
    cout << endl;
    cout << x1 << endl;
    cout << z1 << endl;
    cout << endl;
    cout << rootX1 << endl;
    cout << rootZ1 << endl;
    cout << rootX2 << endl;
    cout << rootZ2 << endl;
    cout << endl;
*/    

    x2 = rootX2;
    z2 = rootZ2;

    double rootY1 = getEllipsoidIntersectionPointsY(rootX1, rootZ1);
    double rootY2 = getEllipsoidIntersectionPointsY(rootX2, rootZ2);

    double l = sqrtf( ((rootX2 - rootX1) * (rootX2 - rootX1)) +
                 ((rootY2 - rootY1) * (rootY2 - rootY1)) +
                 ((rootZ2 - rootZ1) * (rootZ2 - rootZ1)) );

/*
    cout << endl;
    cout << rootX1 << "\t" << rootY1 << "\t" << rootZ1 << "\t:\t";
    cout << rootX2 << "\t" << rootY2 << "\t" << rootZ2 << " = " << l << endl;

    if (l > 0.0) {
        x2 = rootX2;
        z2 = rootZ2;

        x0 = -(c / m);
        y0 = c;
        power = 10.0 * pow(EXP, -l);
    }
*/
    x2 = rootX2;
    z2 = rootZ2;

    //m = getAngleRatio(rootX2, rootZ2, rootX1, rootZ1);

    z0 = 0;
    x0 = getSurfacePointXY(rootZ2, m, rootX2);
    y0 = getSurfacePointXY(rootZ2, m, rootY2);

    //cout << "len = " << l << endl;

    power = pow(EXP, -l);

    //cout << "x0:y0 = " << x0 / GRID_STEP << "\t" << y0 / GRID_STEP << "\t0.0\t" << power << endl;
}

void convertGrid(const vector<vector<struct Point>> &a, vector<vector<struct Point>> &b)
{
    for (auto i: a) {
        for (auto j: i) {
            if ((j.x2 != -1.0) && (j.z2 != -1.0)) {
                double col = (double)j.x2 / GRID_STEP;
                double row = (double)j.z2 / GRID_STEP;

                col = 512.0 + col;
                row = 512.0 + row;

                b[(int)col][(int)row].type  = RAYSHADOW;
                b[(int)col][(int)row].power = 0.0;
            }
        }
    }
}