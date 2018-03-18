/* 
 * udpclient.c - A simple UDP client
 * usage: udpclient <host> <port>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 

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

void error(char *msg) 
{
    perror(msg);
    exit(0);
}

int send_with_reply(int sockfd, struct sockaddr *serveraddr, int serverlen, int part_msg)
{
    protocol_t msg, recv_msg;

    if (part_msg == -1) {
        msg.type = UDP_MESSAGE_COMPLETE;
    } else {
        msg.type = UDP_MESSAGE_SEND;
    }

    sprintf(msg.buffer, "test message, part: %d", part_msg);

    if (sendto(sockfd, &msg, sizeof(protocol_t), 0, serveraddr, serverlen) < 0) {
        error("ERROR in sendto");
    }
    
    if (recvfrom(sockfd, &recv_msg, sizeof(protocol_t), 0, serveraddr, &serverlen) < 0) {
        error("ERROR in recvfrom");
    }
    
    if (recv_msg.type == UDP_MESSAGE_REPLY) {
        //printf("Echo from server: got\n");
        return 0;
    } else {
        //printf("Echo from server: not got {%d|%s}\n", recv_msg.type, recv_msg.buffer);
        return 1;   
    }
}

int main(int argc, char **argv) 
{
    int sockfd, portno, n;
    int serverlen;
    struct sockaddr_in serveraddr;
    struct hostent *server;
    char *hostname;
    char buf[BUFSIZE];

    /* check command line arguments */
    if (argc != 3) {
       fprintf(stderr,"usage: %s <hostname> <port>\n", argv[0]);
       exit(0);
    }
    hostname = argv[1];
    portno = atoi(argv[2]);

    /* socket: create the socket */
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) 
        error("ERROR opening socket");

    /* gethostbyname: get the server's DNS entry */
    server = gethostbyname(hostname);
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host as %s\n", hostname);
        exit(0);
    }

    /* build the server's Internet address */
    bzero((char *) &serveraddr, sizeof(serveraddr));
    serveraddr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, (char *)&serveraddr.sin_addr.s_addr, server->h_length);
    serveraddr.sin_port = htons(portno);

    serverlen = sizeof(serveraddr);

    int part_msg = 0;

    while(1) {

        while(1) {
            if (send_with_reply(sockfd, (struct sockaddr *) &serveraddr, serverlen, part_msg) == 0) {
                printf("send: ok\n");
                break;
            }
            printf("send: fail\n");
        }

        part_msg += 1;
        
        while(1) {
            if (send_with_reply(sockfd, (struct sockaddr *) &serveraddr, serverlen, part_msg) == 0) {
                printf("send: ok\n");
                break;
            }
            printf("send: fail\n");
        }

        while(1) {
            if (send_with_reply(sockfd, (struct sockaddr *) &serveraddr, serverlen, -1) == 0) {
                printf("send: ok\n");
                break;
            }
            printf("send: fail\n");
        }
        
        part_msg = 10;

        sleep(5);
    }
    return 0;
}