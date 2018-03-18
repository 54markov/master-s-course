#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

int get_bit(char byte, int bit)
{
    return((byte >> 8 - bit) & 1);
}

/*
 * Actual image bytes are going form image[54] to image[N-1],
 * so there we are going to inject our message, changing the last bits
 */
void inject_message(char *buffer, char *message)
{
    int cnt = 54;
    
    buffer[cnt] = strlen(message);

    cnt += 4;

    /* Loop by secret message[0..size] */
    for (int i = 0; i < strlen(message); i++) {
        /* Loop by each bit of the message[i] */
        for (int j = 1; j <= 8; j++) {

            int file_byte_lsb = buffer[cnt] & 1;
            int bit_of_message = get_bit(message[i], j);

            if (file_byte_lsb == bit_of_message) {
                // do nothing
            } else {
                if (file_byte_lsb == 0)
                    buffer[cnt] = (buffer[cnt] | 1);
                else
                    buffer[cnt] = (buffer[cnt] & ~1);
            }
            cnt += 4;
        }
    }
}

void collect_message(char *buffer)
{
    int cnt = 54;
    int length = buffer[cnt];

    cnt += 4;

    printf("Dencrypted        : ");

    /* Grab LSB of all bytes for length specified at fgetc */
    for (int i = 0; i < length; i++) {
        char temp_ch = '\0';
        for (int j = 0; j < 8; j++) {
            int file_byte_lsb = buffer[cnt] & 1;
            temp_ch = temp_ch << 1;
            temp_ch |= file_byte_lsb;
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
    if (argc < 1) {
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