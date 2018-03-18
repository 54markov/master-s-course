#include "multi_server.h"

int tcp_fd; // tcp descriptor
int udp_fd; // udp descriptor

void close_server(int sig)
{
    printf("--Closing tcp/udp server--\n");
    close(tcp_fd);
    close(udp_fd);
    exit(0);
}

void thread_handler(int sig)
{
   pid_t tid = gettid();
   printf("Tid %d exited\n", tid);
}

int cretate_udp_soket(int udp_fd, struct sockaddr_in server_addr)
{
    // create UDP socket
    udp_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_fd < 0) {
        fprintf(stderr, "ERROR: socket(udp)\n");
        return -1;
    }

    // bind UDP host address
    if (bind(udp_fd,(struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        fprintf(stderr, "ERROR: bind(udp)\n");
        return -1;
    }

    return udp_fd;
}

int cretate_tcp_socket(int tcp_fd, struct sockaddr_in server_addr)
{
    // create TCP socket
    tcp_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (tcp_fd < 0) {
        fprintf(stderr, "ERROR: socket(tcp)\n");
        return -1;
    }

    // bind TCP host address
    if (bind(tcp_fd,(struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) 
    {
        fprintf(stderr, "ERROR: bind(tcp)\n");
        return -1;
    }
    return tcp_fd;
}

int main( int argc, char *argv[])
{
    char info[256];
    int optval = 1;// options for change flags of socket


    int                 max_fd;// max descritor
    int                 retval;// number connected clients


    fd_set              rset;// main listening socket
    socklen_t           client_len;// client size

    struct timeval      tv;// structure time    
    struct sockaddr_in  server_addr;// structure server
    struct sockaddr_in  client_addr;// structure client

    signal(SIGINT, close_server);// custom interuption signal
    signal(SIGUSR1, thread_handler);// custom interuption signal

    // initialize TCP/UDP socket structure
    bzero((char*)&server_addr, sizeof(server_addr));
    server_addr.sin_family      = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port        = 0;

    // create TCP socket
    tcp_fd = cretate_tcp_socket(tcp_fd, server_addr);
    if (tcp_fd == -1) {
        close_server(1);
    }

    // allow other socket to bind() on this port
    setsockopt(tcp_fd, SOL_SOCKET, SO_REUSEADDR,  &optval, sizeof(optval));

    int lenght = sizeof(server_addr);
    if (getsockname(tcp_fd, (struct sockaddr *) &server_addr, &lenght)) {
        fprintf(stderr, "ERROR: getsockname is failed\n");
        return 1;
    }

    // create UDP socket
    udp_fd = cretate_udp_soket(udp_fd, server_addr);
    if (udp_fd == -1) {
        close_server(1);
    }

    // start listening TCP socket for the clients
    if (listen(tcp_fd, 1000) < 0) {
        fprintf(stderr, "error: listen()");
        close_server(1);
    }

    // Initialize mutex
    if (pthread_mutex_init(&file_mutex, NULL) != 0) {
        fprintf(stderr,"ERROR: file_mutex init failed\n");
        close_server(1);
    }

    printf("\n--Server TCP/UDP established on: [%d]--\n", ntohs(server_addr.sin_port));
    
    sprintf(info, "netstat -tan | grep %d", ntohs(server_addr.sin_port));
    system(info);

    bzero(info, 256);

    sprintf(info, "netstat -uan | grep %d", ntohs(server_addr.sin_port));
    system(info);

    printf("\n--Awaiting connection...--\n");

    max_fd = MAX(tcp_fd, udp_fd);
    client_len = sizeof(client_addr);

    while(1) {
        // set up timeout
        tv.tv_sec = 5;
        tv.tv_usec = 0;

        // set up main socket
        FD_ZERO(&rset);
        FD_SET(tcp_fd, &rset);
        FD_SET(udp_fd, &rset);

        // wait connection
        int retval = select(max_fd+1, &rset, NULL, NULL, &tv);
        if (retval == -1) {
            fprintf(stderr, "Error: select()\n");
            close_server(1);
        } else if (retval == 0) {
            // if timeout do happend and no clients, repeat сycle
            continue;
        }

        // tcp connection
        if (FD_ISSET(tcp_fd, &rset)) {
            int newsockfd = accept(tcp_fd, (struct sockaddr *)&client_addr, &client_len);

            if (newsockfd < 0) {
                fprintf(stderr, "ERROR: on accept\n");
                continue;
            }

            printf("MAIN: %d\n", (int)gettid());
        
            pthread_t *tcp_server_thread = (pthread_t *)malloc(sizeof (pthread_t));
            if (!tcp_server_thread) {
                fprintf(stderr, "ERROR: on malloc\n");
                continue;
            }

            thread_param_t *thread_param = (thread_param_t *)malloc(sizeof (thread_param_t));
            if (!thread_param) {
                fprintf(stderr, "ERROR: on malloc\n");
                continue;
            }

            thread_param->sock_fd  = newsockfd;
            thread_param->cli_addr = client_addr;

            int rc = pthread_create(tcp_server_thread, NULL, &doprocessing, (void *)thread_param);
            if (rc != 0) {
                fprintf(stderr, "ERROR: on pthread_create\n");
                free(thread_param);
                free(tcp_server_thread);
                continue;
            }
        }

        // udp connection
        if (FD_ISSET(udp_fd, &rset)) {
            doprocessing_udp(udp_fd);
        }

    } /* end of while */

    pthread_mutex_destroy(&file_mutex);
    return 0;
}