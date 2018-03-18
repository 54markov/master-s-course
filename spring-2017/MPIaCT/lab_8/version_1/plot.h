#ifndef _PLOT_
#define _PLOT_

#include <png.h>

#include <vector>

void setRGB(png_byte *ptr, double val);

int printRayGrid(char* filename, 
                 int width,
                 int height,
                 const std::vector<std::vector<struct Point>> &grid, 
                 char* title);

#endif /* _PLOT_ */
