#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>


int main(int argc, char const *argv[])
{
    int sockfd;
    int portno;
    int bytesReceived;
    char recvBuff[256] = { 0 };
    char hostname[256];
    char filename[256];
    struct hostent *server;
    struct sockaddr_in serv_addr;

    /* check command line arguments */
    if (argc != 4) {
       fprintf(stderr,"usage: %s <hostname> <port> <file>\n", argv[0]);
       return 1;
    }

    strcpy(hostname, argv[1]);
    strcpy(filename, argv[3]);
    portno   = atoi(argv[2]);

    /* Create a socket first */
    if((sockfd = socket(AF_INET, SOCK_STREAM, 0))< 0) {
        fprintf(stderr, "ERROR: Could not create socket\n");
        return 1;
    }

    /* gethostbyname: get the server's DNS entry */
    server = gethostbyname(hostname);
    if (server == NULL) {
        fprintf(stderr, "ERROR, no such host as %s\n", hostname);
        return 1;
    }

    /* build the server's Internet address */
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr, server->h_length);
    serv_addr.sin_port = htons(portno);

    /* Attempt a connection */
    if(connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr))<0) {
        fprintf(stderr, "Error: Can't connect to server\n");
        return -1;
    }

    /* Receive data in chunks of 256 bytes */
    if (read(sockfd, recvBuff, 256) > 0) {
        printf("Start sending file: %s\n", filename);    
    }

    int serverlen = sizeof(serv_addr);
    if (sendto(sockfd, filename, strlen(filename), 0, (struct sockaddr *)&serv_addr, serverlen) < 0) {
        fprintf(stderr, "ERROR: can't send file name\n");
        close(sockfd);
        return -1;
    }

    /* Open the file that we wish to transfer */
    FILE *fp = fopen(filename,"rb");
    if(!fp) {
        fprintf(stderr, "Error, file open\n");
        close(sockfd);
        return -1;
    }   

    /* Read data from file and send it */
    while(1) {
        /* First read file in chunks of 256 bytes */
        unsigned char buff[256] = {0};
        int nread = fread(buff, 1, 256, fp);
        printf("Bytes read %d \n", nread);        

        /* If read was success, send data. */
        if(nread > 0) {
            printf("Sending %s\n", buff);
            write(sockfd, buff, nread);
        }

        if (nread < 256) {
            if (feof(fp)) {
                printf("End of file\n");
            }
            if (ferror(fp)) {
                printf("Error reading\n");
            }
            break;
        }

    }
    fclose(fp);
    close(sockfd);
    return 0;
}