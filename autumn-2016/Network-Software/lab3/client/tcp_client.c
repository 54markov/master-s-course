#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <pthread.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <unistd.h>
#include <arpa/inet.h>

typedef struct
{
    int socket_fd;
    struct sockaddr_in server_addr;
} client_t;

client_t tcp_client;

void close_client(int sig)
{
    printf("\n--close client--\n");
    close(tcp_client.socket_fd);
    exit(0);
}

int create_socket()
{
    tcp_client.socket_fd = socket(AF_INET, SOCK_STREAM, 0);

    if (tcp_client.socket_fd < 0) {
        fprintf(stderr, "ERROR: can't socket()\n");
        return 1;
    }

    return 0;
}

void fill_socket(char *hostname, int port)
{
    bzero((char*)&tcp_client.server_addr, sizeof(tcp_client.server_addr));

    tcp_client.server_addr.sin_family = AF_INET;
    
    // store this IP address
    inet_pton(AF_INET, hostname, &(tcp_client.server_addr.sin_addr));

    tcp_client.server_addr.sin_port = htons(port);
}

int connet_to_server()
{
    if (connect(tcp_client.socket_fd,(struct sockaddr*)&tcp_client.server_addr, sizeof(tcp_client.server_addr)) < 0) 
    {
        fprintf(stderr, "ERROR: can't connect to server\n");
        return 1;
    }

    return 0;
}

int main(int argc, char const *argv[])
{
    int port = 0;
    int rw_bytes = 0;

    char hostname[256] = { 0 };
    char buffer[256] = { 0 };

    /* Check command line arguments */
    if (argc != 3) {
       fprintf(stderr,"usage: %s <hostname> <port>\n", argv[0]);
       return 1;
    }

    strcpy(hostname, argv[1]);
    port = atoi(argv[2]);

    /* Initialize signal handlers */
    signal(SIGINT, close_client);

    if (create_socket() != 0) {
        return 1;
    }

    fill_socket(hostname, port);

    if (connet_to_server() == 1) {
        close_client(0);
    }

    /********************************************************************/
    /* Start session with server                                        */
    /********************************************************************/

    printf("\n--Start session with server--\n");

    if (read(tcp_client.socket_fd, buffer, 1) < 0) {
        fprintf(stderr,"ERROR: can't read response form server\n");
        close_client(0);
    }

    for (int i = 0; i < 5; i++) {
        bzero(buffer, 256);
        sprintf(buffer, "This is <%d> from test clinet", i);

        if (write(tcp_client.socket_fd, buffer, strlen(buffer)) < 0) {
            fprintf(stderr,"ERROR: can't send to server\n");
            close_client(0);
        }

        printf("send\n");

        if (read(tcp_client.socket_fd, buffer, 1) < 0) {
            fprintf(stderr,"ERROR: can't read response form server\n");
            close_client(0);
        }

        //sleep(5);

        printf("received\n");
    }

    close_client(0);
    return 0;
}