#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 


int socket_fd;

void close_client(int sig)
{
    printf("--Close udp-client--\n");
    close(socket_fd);
    exit(0);
}

void send_with_reply(struct sockaddr *serveraddr, int serverlen)
{
    char buffer[256] = "test message from udp-client";

    int part = 0;

    while (1) {
        if (sendto(socket_fd, &buffer, 256, 0, serveraddr, serverlen) < 0) {
            fprintf(stderr, "ERROR in sendto\n");
            close_client(1);
            return;
        }
    
        int rc = recvfrom(socket_fd, &buffer, 1, 0, serveraddr, &serverlen);
        if (rc > 0){
            printf("part of message sucsessfully send\n");
            part++;
        } else {
            printf("message not send\n");
        }

        if (part == 5) {
            break;
        }
    }

    printf("all message sucsessfully send\n");
    return;
}

int main(int argc, char **argv) 
{
    int portno;
    int serverlen;

    char *hostname;

    struct sockaddr_in serveraddr;
    struct hostent *server;

    struct timeval timeout;      
    
    timeout.tv_sec  = 5;
    timeout.tv_usec = 0;

    /* check command line arguments */
    if (argc != 3) {
       fprintf(stderr,"usage: %s <hostname> <port>\n", argv[0]);
       exit(0);
    }
    hostname = argv[1];
    portno = atoi(argv[2]);

    signal(SIGINT, close_client);

    /* socket: create the socket */
    socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd < 0) {
        fprintf(stderr,"ERROR opening socket\n");
        return 1;
    }

    /* gethostbyname: get the server's DNS entry */
    server = gethostbyname(hostname);
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host as %s\n", hostname);
        return 1;
    }

    /* build the server's Internet address */
    bzero((char *) &serveraddr, sizeof(serveraddr));
    bcopy((char *)server->h_addr, (char *)&serveraddr.sin_addr.s_addr, server->h_length);
    
    serveraddr.sin_family = AF_INET;
    serveraddr.sin_port   = htons(portno);

    if (setsockopt (socket_fd, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout)) < 0) {
        fprintf(stderr,"ERROR: setsockopt failed (SO_RCVTIMEO)\n");
        close_client(1);
    }

    if (setsockopt (socket_fd, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeout, sizeof(timeout)) < 0) {
        fprintf(stderr,"ERROR: setsockopt failed (SO_SNDTIMEO)\n");
        close_client(1);
    }

    serverlen = sizeof(serveraddr);

    while(1) {
        send_with_reply((struct sockaddr *)&serveraddr, serverlen);
        sleep(2);
    }

    close_client(1);
    return 0;
}