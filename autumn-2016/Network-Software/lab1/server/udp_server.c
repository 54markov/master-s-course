/* 
 * udpserver.c - A simple UDP echo server 
 * usage: udpserver <port>
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "queue.h"

#define COLOR_RED "\033[31m"
#define COLOR_T   "\033[33m"
#define COLOR_BLU "\033[34m"
#define COLOR_OFF "\033[0m"

#define BUFSIZE 1024

enum
{
    UDP_MESSAGE_UNKNOWN  = 0,
    UDP_MESSAGE_SEND     = 1,
    UDP_MESSAGE_COMPLETE = 2,
    UDP_MESSAGE_REPLY    = 3
};

typedef struct UDP_REPLY_PROTOCOL
{
    int type;
    char buffer[BUFSIZE];
} protocol_t;

void error(char *msg) {
    perror(msg);
    exit(1);
}

int main(int argc, char **argv) 
{
    int sockfd;                     /* socket */
    int lenght;
    int clientlen;                  /* byte size of client's address */
    struct sockaddr_in serveraddr;  /* server's addr */
    struct sockaddr_in clientaddr;  /* client addr */
    struct hostent *hostp;          /* client host info */
    char buf[BUFSIZE];              /* message buf */
    char *hostaddrp;                /* dotted decimal host addr string */
    int optval;                     /* flag value for setsockopt */
    int n;                          /* message byte size */

    if (argc < 2) {
        fprintf(stderr, "usage: lost packet\n");
        return 0;
    }

    int param = atoi(argv[1]);

    /* socket: create the parent socket */
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        error("ERROR opening socket");
    }

    /* build the server's Internet address */
    bzero((char *) &serveraddr, sizeof(serveraddr));

    serveraddr.sin_family      = AF_INET;
    serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
    serveraddr.sin_port        = 0;

    /* bind: associate the parent socket with a port */
    if (bind(sockfd, (struct sockaddr *) &serveraddr, sizeof(serveraddr)) < 0) {
        error("ERROR on binding");
    }

    lenght = sizeof(serveraddr);
    if (getsockname(sockfd, (struct sockaddr *) &serveraddr, &lenght)) {
        error("Call getsockname is failed\n");
        exit(1);
    }

    printf("Server: prot number <%d>\n", ntohs(serveraddr.sin_port));

    /* main loop: wait for a datagram, then echo it */
    clientlen = sizeof(clientaddr);

    int count = 1;

    queue_init();

    while(1) {
        /* recvfrom: receive a UDP datagram from a client */
        bzero(buf, BUFSIZE);
        if (recvfrom(sockfd, buf, BUFSIZE, 0,(struct sockaddr *) &clientaddr, &clientlen) < 0) {
            error("ERROR in recvfrom");
        }

        /* 
         * gethostbyaddr: determine who sent the datagram
         */
        hostp = gethostbyaddr((const char *)&clientaddr.sin_addr.s_addr, sizeof(clientaddr.sin_addr.s_addr), AF_INET);
        if (!hostp) {
            error("ERROR on gethostbyaddr");
        }

        hostaddrp = inet_ntoa(clientaddr.sin_addr);
        if (!hostaddrp) {
            error("ERROR on inet_ntoa\n");
        }

        //printf("server received datagram from %s (%s) port %d\n", hostp->h_name, hostaddrp, clientaddr.sin_port);

        protocol_t *recv_msg = (protocol_t *) &buf;
        protocol_t msg;

        switch (recv_msg->type)
        {
            case UDP_MESSAGE_SEND:
                if (count % param == 0) {
                    msg.type = UDP_MESSAGE_REPLY;
                    printf(COLOR_BLU);
                    printf("server received %s (%s) port %d %d/%d bytes: %s\n", hostp->h_name, 
                                                                                hostaddrp, 
                                                                                clientaddr.sin_port, 
                                                                                (int)strlen(recv_msg->buffer), 
                                                                                n, 
                                                                                recv_msg->buffer);
                    printf(COLOR_OFF);

                    add_to_queue(clientaddr.sin_port, recv_msg->buffer);

                } else {
                    msg.type = UDP_MESSAGE_UNKNOWN;
                    printf(COLOR_RED);
                    printf("server received %s (%s) port %d %d/%d bytes: %s\n", hostp->h_name, 
                                                                            hostaddrp, 
                                                                            clientaddr.sin_port, 
                                                                            (int)strlen(recv_msg->buffer), 
                                                                            n, 
                                                                            recv_msg->buffer);
                    printf(COLOR_OFF);
                }
                count += 1;

                if (sendto(sockfd, &msg, sizeof(protocol_t), 0, (struct sockaddr *) &clientaddr, clientlen) < 0) {
                    error("ERROR in sendto");
                }
                break;

            case UDP_MESSAGE_COMPLETE:
                printf(COLOR_T);
                printf("server received %s (%s) port %d %d/%d bytes\n", hostp->h_name, 
                                                                            hostaddrp, 
                                                                            clientaddr.sin_port, 
                                                                            (int)strlen(recv_msg->buffer), 
                                                                            n);
                printf(COLOR_OFF);

                free_queue(clientaddr.sin_port);

                if (sendto(sockfd, &msg, sizeof(protocol_t), 0, (struct sockaddr *) &clientaddr, clientlen) < 0) {
                    error("ERROR in sendto");
                }
                break;

            default:
                fprintf(stderr, "UNKNOWN type: %d\n", recv_msg->type);
                break;
        }
    }
}