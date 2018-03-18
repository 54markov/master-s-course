#include <iostream>
#include <vector>
#include <cmath>

#include "main.h"
#include "plot.h"
#include "point.h"

#include "Colors.h"

using namespace std;

void initilizeRayShadowGrid(std::vector<std::vector<struct Point>> &grid)
{
    for (double xi = -GRID; xi < 2.0*GRID; xi = xi + GRID_STEP) {
        std::vector<struct Point> newRow;
        for (double yi = -GRID; yi < 2.0*GRID; yi = yi + GRID_STEP) {
            struct Point P;
            newRow.push_back(P);
        }
        grid.push_back(newRow);
    }
}

int main(int argc, char const *argv[])
{
    char filename[256] = { "result_pic1.png" };
    char title[256]    = { "ray_ellipsoid1" };

    char filename1[256] = { "result_pic2.png" };
    char title1[256]    = { "ray_ellipsoid2" };

    std::vector<std::vector<struct Point>> projectionGridByXY;
    std::vector<std::vector<struct Point>> rayShadowGrid;

    initilizeRayShadowGrid(rayShadowGrid);

    /*
     * From = -1.0
     * TO   = +1.0
     * STEP = 0.00195312
     */
    int col = 0;
    for (double xi = -GRID; xi < GRID; xi = xi + GRID_STEP) {
        
        std::vector<struct Point> newRow;
        int row = 0;
        for (double yi = -GRID; yi < GRID; yi = yi + GRID_STEP) {
            struct Point newPoint;

            /* Initilize upper ellipsiod intersection point */
            newPoint.x1 = xi;
            newPoint.y1 = yi;
            newPoint.z1 = -1.0;
            
            /* Initilize lower ellipsiod intersection point */
            newPoint.x2 = -1.0;
            newPoint.y2 = -1.0;
            newPoint.z2 = -1.0;
            
            /* Initilize surface x,y,z intersection point */
            newPoint.x0 = -1.0;
            newPoint.y0 = -1.0;
            newPoint.z0 = -1.0;

            /* Initilize type of the point IN/OUT -side or INTERSECTION */
            newPoint.type = getEllipsoidIntersectionPoints(xi, yi);

            /* Initilize pay power of the point */
            newPoint.power = -1.0;


            if (newPoint.type != OUTSIDE) {
                newPoint.z1 = getEllipsoidIntersectionPointsZ(xi, yi);
                getSecondPoints(newPoint.x2,
                                newPoint.z2,
                                newPoint.power,
                                newPoint.x0,
                                newPoint.y0,
                                newPoint.z0,
                                newPoint.x1,
                                newPoint.z1);
            }
            
            if ((newPoint.x2 != -1.0) && (newPoint.z2 != -1.0)) {
                newPoint.type  = RAYSHADOW;
                newPoint.y2    = getEllipsoidIntersectionPointsY(newPoint.x2, newPoint.z2);

                //double row = newPoint.x0 / GRID_STEP;
                //double col = newPoint.y0 / GRID_STEP;

                //col = xi;
                //row = yi;

                double m = getAngleRatio(0.0, 5.0, newPoint.x2, newPoint.z2);
                m = atan(m);
/*
                //cout << endl;
                cout << "m = " << m << endl;
                cout << "x2:y2 = " << newPoint.x2 << "\t" << newPoint.y2 << "\t" << newPoint.power << endl;
                cout << "x0:y0 = " << newPoint.x0 << "\t" << newPoint.y0 << "\t" << newPoint.power << endl;
                //cout << "col:row = " << col << "\t" << row << endl;
*/

/*                
                if (abs(col) > 1024) {
                    cerr << __FUNCTION__ << "Error: col greater " << col << endl;
                }
                if (abs(row) > 1024) {
                    cerr << __FUNCTION__ << "Error: col greater " << col << endl; 
                }
*/

                //rayShadowGrid[col][row].power = 0.99;
                //rayShadowGrid[col][row].type  = newPoint.type;

                double localCol = 0.0;
                double localRow = 0.0;

                double tcol = m * (double)col;
                double trow = m * (double)row;

                if (tcol < 0.0) {
                    localCol = col + (abs(tcol) - col);
                } else {
                    localCol = tcol;
                }

                if (trow < 0.0) {
                    localRow = row + (abs(trow) - row);
                } else { 
                    localRow = trow;
                }
/*
                cout << "\t" << tcol << " - " << trow << endl;

                cout << col << " - " << localCol << endl;
                cout << row << " - " << localRow << endl;
*/
                rayShadowGrid[localCol][localRow] = newPoint;
            } else {
                //rayShadowGrid[col][row] = newPoint;
            }
            row++;
            newRow.push_back(newPoint);
        }
        col++;
        projectionGridByXY.push_back(newRow);
    }

    //cout << projectionGridByXY.size() << endl;

    printRayGrid(filename, projectionGridByXY.size(), 
                    projectionGridByXY.size(), projectionGridByXY, title);

    printRayGrid(filename1, rayShadowGrid.size(), 
                    rayShadowGrid.size(), rayShadowGrid, title1);

    return 0;
}
