#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "main.h"
#include "plot.h"
#include "point.h"

#include "Colors.h"

using namespace std;

int main(int argc, char const *argv[])
{
    char filename[256] = { "xray.png" };
    char filename1[256] = { "xray_noise.png" };
    char filename2[256] = { "xray_noise_filter.png" };
    char title[256]    = { "xray" };

    std::vector<std::vector<struct Point>> projectionGridByXY;

    /*
     * From = -1.0
     * TO   = +1.0
     * STEP = 0.00195312
     */
    for (double xi = -GRID; xi < GRID; xi = xi + GRID_STEP) {
        
        std::vector<struct Point> newRow;
        for (double yi = -GRID; yi < GRID; yi = yi + GRID_STEP) {
            struct Point newPoint;
            newPoint.power = getSurfacePoints(xi, yi);
            newRow.push_back(newPoint);
        }
        projectionGridByXY.push_back(newRow);
    }

    printRayGrid(filename, projectionGridByXY.size(),
                    projectionGridByXY.size(), projectionGridByXY, title, 0);

    printRayGrid(filename1, projectionGridByXY.size(),
                    projectionGridByXY.size(), projectionGridByXY, title, 1);

    printRayGrid1(filename2, projectionGridByXY.size(),
                    projectionGridByXY.size(), projectionGridByXY, title, 1);

    return 0;
}
