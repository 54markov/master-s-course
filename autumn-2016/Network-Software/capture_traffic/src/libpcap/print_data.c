#include "header.h"

void print_data(const u_char *data , int Size)
{
    int i, j;

    for (i = 0; i < Size; i++) {
        if (i != 0 && i%16 == 0) {  //if one line of hex printing is complete...
            printf("         ");
            for (j = i - 16; j < i; j++) {
                if (data[j] >= 32 && data[j] <= 128) {
                    printf("%c", (unsigned char)data[j]); //if its a number or alphabet
                } else {
                    printf("."); //otherwise print a dot
                }
            }
            printf("\n");
        } 
         
        if (i%16 == 0) { 
            printf("   ");
        }

        printf(" %02X", (unsigned int)data[i]);
                 
        if (i == Size - 1) {  //print the last spaces
            for (j = 0; j < 15 - i%16; j++) {
                printf("   "); //extra spaces
            }
            printf("         ");             
            for (j = i - i%16; j <= i; j++) {
                if (data[j] >= 32 && data[j] <= 128) {
                    printf("%c", (unsigned char)data[j]);
                }
                else {
                    printf(".");
                }
            }
            printf("\n");
        }
    }

    char *ptr = NULL;

    ptr = strstr((const char*) data, (const char*)word);

    if (ptr != NULL) {
        printf(COLOR_RED);
        printf("\n%s\n", ptr);
        printf(COLOR_OFF);
    }
}
