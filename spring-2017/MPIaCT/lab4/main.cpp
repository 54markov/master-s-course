#include "Field.h"

#include <sys/time.h>
#include <stdio.h>

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

int main(int argc, char const *argv[])
{
    /* Warm-up */
    printf("Warm-up: start\n");
    Field field;
    field.generateWayPoints(3);
    field.getPathBetweenWayPoints();
    printf("Warm-up: end\n");

    for (auto i = 3; i <= 11; i++) {
        auto t = wtime();
        for (auto j = 0; j < 10; j++) {
            Field field;
            field.generateWayPoints(i);
            //field.printField();
            field.getPathBetweenWayPoints();
        }
        t = wtime() - t;
        printf("Elapsed time (%d): %.6f sec.\n", i, t / 10);
    }
    return 0;
}