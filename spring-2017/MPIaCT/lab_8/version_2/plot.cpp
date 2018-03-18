#include "plot.h"
#include "point.h"
#include "main.h"

#include <iostream>
#include <vector>
#include <random>

using namespace std;

static void setRGB(png_byte *ptr, double power)
{
    if (power == -1) {
        ptr[0] = 0; ptr[1] = 0; ptr[2] = 0; // black color
        return;
    }
/*
    ptr[0] = (int)(200.0 * 5.0 * (0.5 - power));
    ptr[1] = (int)(200.0 * 5.0 * (0.5 - power));
    ptr[2] = (int)(200.0 * 5.0 * (0.5 - power));
*/

    ptr[0] = (int)(200.0 * 5.0 * (1.0 - power));
    ptr[1] = (int)(200.0 * 5.0 * (1.0 - power));
    ptr[2] = (int)(200.0 * 5.0 * (1.0 - power));

}

int printRayGrid(char* filename, 
                 int width,
                 int height,
                 const vector<vector<struct Point>> &grid, 
                 char* title,
                 int noise_mode)
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
        goto finalize;
    }

    // Initialize write structure
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
        fprintf(stderr, "Could not allocate write struct\n");
        code = 1;
        goto finalize;
    }

    // Initialize info structure
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
        fprintf(stderr, "Could not allocate info struct\n");
        code = 1;
        goto finalize;
    }

    // Setup Exception handling
    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error during png creation\n");
        code = 1;
        goto finalize;
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
    if (noise_mode) {
        int x;
        for (auto i: grid) {
            x = 0;
            for (auto j: i) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(0.0, 0.20);
                if (j.power != -1.0) {
                    j.power += 1.5 * (dis(gen) - 0.01);
                }
                setRGB(&(row[x*3]), j.power);
                x++;
            }
            png_write_row(png_ptr, row);
        }
    } else {
        int x;
        for (auto i: grid) {
            x = 0;
            for (auto j: i) {
                setRGB(&(row[x*3]), j.power);
                x++;
            }
            png_write_row(png_ptr, row);
        }
    }

    // End write
    png_write_end(png_ptr, NULL);

finalize:
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


int printRayGrid1(char* filename, 
                 int width,
                 int height,
                 const vector<vector<struct Point>> &grid, 
                 char* title,
                 int noise_mode)
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
        goto finalize;
    }

    // Initialize write structure
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
        fprintf(stderr, "Could not allocate write struct\n");
        code = 1;
        goto finalize;
    }

    // Initialize info structure
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
        fprintf(stderr, "Could not allocate info struct\n");
        code = 1;
        goto finalize;
    }

    // Setup Exception handling
    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error during png creation\n");
        code = 1;
        goto finalize;
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
    if (noise_mode) {
        int x;
        for (auto i: grid) {
            x = 0;
            for (auto j: i) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(0.0, 0.1);
                if (j.power != -1.0) {
                    j.power += 1.5 * (dis(gen));
                }
                setRGB(&(row[x*3]), j.power);
                x++;
            }
            png_write_row(png_ptr, row);
        }
    } else {
        int x;
        for (auto i: grid) {
            x = 0;
            for (auto j: i) {
                setRGB(&(row[x*3]), j.power);
                x++;
            }
            png_write_row(png_ptr, row);
        }
    }

    // End write
    png_write_end(png_ptr, NULL);

finalize:
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

double make_filtration(int x, int y, const vector<vector<struct Point>> &grid)
{
    double top = grid[x-1][y];
    double botom = grid[x+1][y];
    double left = grid[x][y-1];
    double right = grid[x][y+1];

    return (top + botom + left + right) / 4.0;

}

int printRayGrid2(char* filename, 
                 int width,
                 int height,
                 const vector<vector<struct Point>> &grid, 
                 char* title,
                 int noise_mode)
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
        goto finalize;
    }

    // Initialize write structure
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
        fprintf(stderr, "Could not allocate write struct\n");
        code = 1;
        goto finalize;
    }

    // Initialize info structure
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
        fprintf(stderr, "Could not allocate info struct\n");
        code = 1;
        goto finalize;
    }

    // Setup Exception handling
    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error during png creation\n");
        code = 1;
        goto finalize;
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
    int row = 0;
    if (noise_mode) {
        int x;
        for (auto i: grid) {
            x = 0;
            for (auto j: i) {

                j.power = make_filtration(row, x);
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(0.0, 0.2);
                j.power += 1.5 * (dis(gen));
                setRGB(&(row[x*3]), j.power);
                x++;
            }
            row++;
            png_write_row(png_ptr, row);
        }
    } else {
        int x;
        for (auto i: grid) {
            x = 0;
            for (auto j: i) {
                setRGB(&(row[x*3]), j.power);
                x++;
            }
            png_write_row(png_ptr, row);
        }
    }

    // End write
    png_write_end(png_ptr, NULL);

finalize:
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
