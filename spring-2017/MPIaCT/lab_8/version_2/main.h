#ifndef _MAIN_
#define _MAIN_

#include "point.h"

#define EXP             2.71828182846
#define GRID            1.0
#define GRID_RESOLUTION 1024.0
#define GRID_STEP       0.00195312

enum
{
    INTERSECTION = 0,
    INSIDE       = 1,
    OUTSIDE      = 2,
    RAYSHADOW    = 3
};

#endif /* _MAIN_ */