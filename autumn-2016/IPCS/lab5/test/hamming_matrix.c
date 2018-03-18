#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

unsigned short set_bit1(unsigned short byte, int pos)
{
    byte |= 1 << pos;

    return byte;
}

unsigned short clr_bit1(unsigned short byte, int pos)
{
    byte &= ~(1 << pos);

    return byte;
}

char set_bit(char byte, int pos)
{
    byte |= 1 << pos;

    return byte;
}

char clr_bit(char byte, int pos)
{
    byte &= ~(1 << pos);

    return byte;
}

int get_bit(char byte, int pos)
{
    int bit = (byte >> pos) & 1;

    return bit;
}

int get_hamming_matrix(char buffer[], char ch, int counter)
{
    int c[15];
    int sh[4][15];
    int sl[4][15];

    unsigned short S[4] = { 0 };
    int indexl = 0, indexh = 0;

    int m[8];

    for (int i = 0; i < 8; i++) {
        m[i] = get_bit(ch, i);
    }

    printf("\nM\n");
    for (int i = 0; i < 8; i++) {
        printf("%d ", m[i]);
    }
    printf("\n");

    /* high part */
    for (int j = 0, cnt = counter; j < 15; j++, cnt++) {
        c[j] = get_bit(buffer[cnt], 7);
    }

    printf("\nC\n");
    for (int j = 0; j < 15; j++) {
        printf("%d ", c[j]);
    }
    printf("\n");

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 15; j++) {
            if ((c[j] == 0) && (m[i] == 0)) {
                sh[i][j] = 0;
            } else if ((c[j] == 1) && (m[i] == 1)) {
                sh[i][j] = 0;
            } else {
                sh[i][j] = 1;
            }
        }
    }

    /* get measures */
    printf("\nSH\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 15; j++) {
            printf("%d ", sh[i][j]);
            if (sh[i][j] == 1) {
                S[i] = set_bit1(S[i], j);
            } else {
                S[i] = clr_bit1(S[i], j);
            }
        }
        printf("S: %d\n", S[i]);
    }
    printf("\n");

    for (int i = 0; i < 4; i++) {
        if (S[i] > 15) {
            S[i] = 15;
        }
    }

    indexh = (8 * S[3]) + (4 * S[2]) + (2 * S[1]) + (1 * S[0]);
    printf("indexh: %d\n", indexh / 15);

    if (get_bit(buffer[counter + indexh], 7) == 1) {
        set_bit(buffer[counter + indexh], 7);
    }

    /* low part */

    counter += 15;

    for (int j = 0, cnt = counter; j < 15; j++, cnt++) {
        c[j] = get_bit(buffer[cnt], 7);
    }

    printf("\nC\n");
    for (int j = 0; j < 15; j++) {
        printf("%d ", c[j]);
    }
    printf("\n");

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 15; j++) {
            if ((c[j] == 0) && (m[i] == 0)) {
                sl[i][j] = 0;
            } else if ((c[j] == 1) && (m[i] == 1)) {
                sl[i][j] = 0;
            } else {
                sl[i][j] = 1;
            }
        }
    }

    /* get measures */
    printf("\nSL\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 15; j++) {
            printf("%d ", sl[i][j]);
            if (sl[i][j] == 1) {
                S[i] = set_bit1(S[i], j);
            } else {
                S[i] = clr_bit1(S[i], j);
            }
        }
        printf("S: %d\n", S[i]);
    }
    printf("\n");

    for (int i = 0; i < 4; i++) {
        if (S[i] > 15) {
            S[i] = 15;
        }
    }

    indexl = (8 * S[3]) + (4 * S[2]) + (2 * S[1]) + (1 * S[0]);
    printf("indexl: %d\n", indexl/15);

    if (get_bit(buffer[counter + indexl], 7) == 1) {
        set_bit(buffer[counter + indexl], 7);
    }

    return counter;
}

/*
void get_hamming_message(int *hamming_message, int size, char message)
{
    int j = 0;

    hamming_message[0] = 0;
    hamming_message[1] = 0;

    for (int i = 2; i < size; i++) {
        if (i == 3) {
            hamming_message[i] = 0; i++;
        }

        if (i == 7) {
            hamming_message[i] = 0; i++;
        }

        if (get_bit(message, j) == 0) {
            hamming_message[i] = 0;
        } else {
            hamming_message[i] = 1;
        }
        j++;
    }
}

void get_hamming_code(int *hamming_message, int size)
{
    int control_bits[4] = {1, 2, 4, 8};

    for (int i = 0; i < 4; i++) {
        int value = 0;
        int cnt = control_bits[i];

        for (int j = cnt - 1; j < size; j += cnt) {
            value += hamming_message[j];
        }

        if ((value % 2) == 0) {
            hamming_message[cnt - 1] = 0;
        } else {
            hamming_message[cnt - 1] = 1;
        }

    }
}
*/
/*
 * Actual image bytes are going form image[54] to image[N-1],
 * so there we are going to inject our message, changing the last bits
 */
void inject_message(char *buffer, char *message)
{
    int cnt = 54;

    for (int i = 0; i < strlen(message); i++) {
        cnt = get_hamming_matrix(buffer, message[i], cnt);
    }
/*
        for (int j = 0; j < 16; j++) {
            printf("%d", hamming_message[j]);
        }
        printf("\n");
*/
        //get_hamming_code(hamming_message, 16);
/*
        for (int j = 0; j < 16; j++) {
            printf("%d", hamming_message[j]);
        }
        printf("\n\n");
*/
/*
        for (int j = 0; j < 16; j++) {
            if (hamming_message[j] == 0) {
                buffer[cnt] = clr_bit(buffer[cnt], 7);
            } else {
                buffer[cnt] = set_bit(buffer[cnt], 7);
            }
            cnt += 4;
        }

    }
*/
    return;
/*
    buffer[cnt] = strlen(hamming_message);

    cnt += 4;

    for (int i = 0; i < strlen(hamming_message); i++) {
        if (hamming_message[i] == 0) {
            buffer[cnt] = clr_bit(buffer[cnt], 7);
        } else {
            buffer[cnt] = set_bit(buffer[cnt], 7);
        }
        cnt += 4;
    }
*/
}

void collect_message(char *buffer)
{
    int cnt = 54;
    int length = buffer[cnt];

    cnt += 4;

    printf("Dencrypted        : ");

    for (int i = 0; i < length; i++) {
        char temp_ch = 0;
        for (int j = 0; j < 8; j++) {
            if (get_bit(buffer[cnt], 7) == 0) {
                temp_ch = clr_bit(temp_ch, j);
            } else {
                temp_ch = set_bit(temp_ch, j);
            }
            cnt += 4;
        }
        printf("%c", temp_ch);
    }
    printf("\n");
}

FILE *open_image(const char *file, char *mode)
{
    FILE *fp = fopen(file, mode);
    if (!fp) {
        fprintf(stderr, "Can't open input file %s\n", file);
        return NULL;
    }
    return fp;
}

int main(int argc, char const *argv[])
{
    struct stat file_stat;
    char *image;
    int n, mode;
    FILE *in_fp, *out_fp;
    
    /* Checking the arguments */
    if (argc < 2) {
        fprintf(stderr, "usage: %s -e [original.image] [new.image] [message]\n", argv[0]);
        fprintf(stderr, "usage: %s -d [new.image]\n", argv[0]);
        return 1;
    }

    if(!strcmp(argv[1],"-e")) {
        mode = 1;
        printf("Mode is           : encrypt image\n");
        printf("Original image is : %s\n", argv[2]);
        printf("New image will be : %s\n", argv[3]);
        printf("Secret message is : %s\n", argv[4]);

        /* Open original image */
        if ((in_fp = open_image(argv[2], "rb+")) == NULL) {
            return 1;
        }

        /* Open new image */
        if ((out_fp = open_image(argv[3], "wb+")) == NULL) {
            fclose(in_fp);
            return 1;
        }

        /* Read original image size */
        fstat(fileno(in_fp), &file_stat);
        if (file_stat.st_size < 0) {
            fprintf(stderr, "File size failure %s\n", argv[1]);
            fclose(in_fp);
            fclose(out_fp);
            return(1);
        }

    } else if(!strcmp(argv[1],"-d")) {
        mode = 0;
        printf("Mode is           : dencrypt image\n");
        printf("New image will be : %s\n", argv[2]);

        /* Open original image */
        if ((in_fp = open_image(argv[2], "rb+")) == NULL) {
            return 1;
        }

        /* Read original image size */
        fstat(fileno(in_fp), &file_stat);
        if (file_stat.st_size < 0) {
            fprintf(stderr, "File size failure %s\n", argv[1]);
            fclose(in_fp);
            return(1);
        }

    } else {
        fprintf(stderr, "usage: %s -e [original.image] [new.image] [message]\n", argv[0]);
        fprintf(stderr, "usage: %s -d [new.image]\n", argv[0]);
        return 1;
    }

    /* Allocate memory for image buffer */
    image = (char*)malloc(file_stat.st_size);
    if (!image) {
        fprintf(stderr, "Can't allocate memory\n");
        fclose(in_fp);
        fclose(out_fp);
        return 1;
    }

    if (mode) {
        // Read whole file
        size_t rbytes = fread(image, 1, file_stat.st_size, in_fp);

        inject_message(image, (char *)argv[4]);
        
        // Write whole file
        size_t wbytes = fwrite(image, 1, file_stat.st_size, out_fp);

        if (rbytes != wbytes)
           fprintf(stderr, "error while writing to file\n"); 
        
        fclose(in_fp);
        fclose(out_fp);
    } else {
        // Read whole file
        size_t rbytes = fread(image, 1, file_stat.st_size, in_fp);
        collect_message(image);
        fclose(in_fp);
    }

    free(image);
    return 0;
}