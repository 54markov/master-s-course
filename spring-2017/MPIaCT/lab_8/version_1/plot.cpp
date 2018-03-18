#include "plot.h"
#include "point.h"
#include "main.h"

#include <iostream>
#include <vector>

using namespace std;

void setRGB(png_byte *ptr, double power)
{
    if (power == -1) {
        ptr[0] = 0; ptr[1] = 0; ptr[2] = 0; // black color
        return;
    }
/* 
    if (power > 0.9) {                            // very-high
        ptr[0] = 115; ptr[1] = 115; ptr[2] = 115;
    } else if ((power < 0.9) && (power > 0.8)) {  // high
        ptr[0] = 135; ptr[1] = 135; ptr[2] = 135;
    } else if ((power < 0.8) && (power > 0.7)) {  // high-medium
        ptr[0] = 145; ptr[1] = 145; ptr[2] = 145;
    } else if ((power < 0.7) && (power > 0.6)) {  // medium
        ptr[0] = 165; ptr[1] = 165; ptr[2] = 165;
    } else if ((power < 0.6) && (power > 0.5)) {  // low-medium
        ptr[0] = 185; ptr[1] = 185; ptr[2] = 185;
    } else if ((power < 0.5) && (power > 0.4)) {  // low
        ptr[0] = 200; ptr[1] = 200; ptr[2] = 200;
    } else {                                      // very-low
        ptr[0] = 255; ptr[1] = 255; ptr[2] = 255;
    }
*/  
    int color = (int)(200.0 * 2.15 * (1.0 - power)); 
    ptr[0] = color; 
    ptr[1] = color; 
    ptr[2] = color;
}

int printRayGrid(char* filename, 
                 int width,
                 int height,
                 const vector<vector<struct Point>> &grid, 
                 char* title)
{
    int code            = 0;
    FILE *fp            = NULL;
    png_structp png_ptr = NULL;
    png_infop info_ptr  = NULL;
    png_bytep row       = NULL;
    
    // Open file for writing (binary mode)
    fp = fopen(filename, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Could not open file %s for writing\n", filename);
        code = 1;
        goto finalise;
    }

    // Initialize write structure
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
        fprintf(stderr, "Could not allocate write struct\n");
        code = 1;
        goto finalise;
    }

    // Initialize info structure
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
        fprintf(stderr, "Could not allocate info struct\n");
        code = 1;
        goto finalise;
    }

    // Setup Exception handling
    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error during png creation\n");
        code = 1;
        goto finalise;
    }

    png_init_io(png_ptr, fp);

    // Write header (8 bit colour depth)
    png_set_IHDR(png_ptr, info_ptr, width, height,
            8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    // Set title
    if (title != NULL) {
        png_text title_text;
        title_text.compression = PNG_TEXT_COMPRESSION_NONE;
        title_text.key  = "Title";
        title_text.text = title;
        png_set_text(png_ptr, info_ptr, &title_text, 1);
    }

    png_write_info(png_ptr, info_ptr);

    // Allocate memory for one row (3 bytes per pixel - RGB)
    row = (png_bytep) malloc(3 * width * sizeof(png_byte));

    // Write image data
    int x;
    for (auto i: grid) {
        x = 0;
        for (auto j: i) {
            setRGB(&(row[x*3]), j.power);
            x++;
#ifdef DEBUG
            cout << j.x1  << "\t" << j.y1 << "\t" << j.z1;
            cout << "\t" << j.x0 << "\t" << j.y0 << "\t" << j.z0;
            cout << "\t" << j.power;
            switch((int)j.type)
            {
                case INTERSECTION:
                    cout << "\tINTERSECTION";
                    break;
                case INSIDE:
                    cout << "\tINSIDE";
                    break;
                case OUTSIDE:
                    cout << "\tOUTSIDE";
                    break;
                case RAYSHADOW:
                    cout << "\tRAYSHADOW";
                    break;
            }
            cout << endl;
#endif /* DEBUG */
        }
        png_write_row(png_ptr, row);
#ifdef DEBUG
        cout << endl;
#endif /* DEBUG */
    }
#ifdef DEBUG
    cout << endl;
#endif /* DEBUG */
    // End write
    png_write_end(png_ptr, NULL);

finalise:
    if (fp != NULL) 
        fclose(fp);

    if (info_ptr != NULL) 
        png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);

    if (png_ptr != NULL) 
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);

    if (row != NULL) 
        free(row);

    return code;
}
