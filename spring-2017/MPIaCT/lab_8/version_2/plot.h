#ifndef _PLOT_
#define _PLOT_

#include <vector>

#include <png.h>

int printRayGrid(char* filename, int width, int height,
                 const std::vector<std::vector<struct Point>> &grid, 
                 char* title,
                 int mode);

int printRayGrid1(char* filename, 
                 int width,
                 int height,
                 const std::vector<std::vector<struct Point>> &grid, 
                 char* title,
                 int noise_mode);


#endif /* _PLOT_ */
