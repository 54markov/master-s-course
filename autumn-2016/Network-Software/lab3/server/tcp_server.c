#include "multi_server.h"

void save_protected(char *buf)
{
    pthread_mutex_lock(&file_mutex);

    /* Create file where data will be stored */
    FILE *fp = fopen("LOG", "ab"); 
    if (!fp) {
        fprintf(stderr, "ERROR: opening file\n");
        pthread_mutex_unlock(&file_mutex);
        return ;
    }

    fwrite(buf, 1, strlen(buf), fp);

    fclose(fp);

    pthread_mutex_unlock(&file_mutex);
    return;
}

void *doprocessing(void *args)
{
    int sock_fd;
    thread_param_t *thread_param;
    struct sockaddr_in clientaddr;

    char *hostaddrp = NULL;
    char buf[256]   = { 0 };
    char file_buf[512] = { 0 };

    thread_param = (thread_param_t*)args;
    sock_fd      = thread_param->sock_fd;
    clientaddr   = thread_param->cli_addr;

    printf("Child thread id: %d\n", (int)gettid());

    printf("\n--Start session with client--\n");

    if (write(sock_fd, buf, 1) < 0) {
        fprintf(stderr,"ERROR: can't write response to client\n");
        close(sock_fd);
        printf("\n-- child thread call kill--\n");
        pthread_kill(pthread_self(), SIGUSR1);
        printf("\n--child thread call exit--\n");
        pthread_exit(NULL);
        printf("\n--child thread never happend--\n");
    }

    while(1) {

        bzero(buf, 256);
        bzero(file_buf, 512);

        int rc = read(sock_fd, buf, 255);

        if (rc == 0) {
            break;
        } else if (rc < 0) {
            fprintf(stderr, "ERROR: can't read  from client\n");
            break;
        }

        hostaddrp = inet_ntoa(clientaddr.sin_addr);
        if (!hostaddrp) {
            fprintf(stderr, "ERROR on inet_ntoa\n");
        }
        
        sprintf(file_buf, "received (%s) port %d %d bytes: %s\n", hostaddrp, clientaddr.sin_port, (int)strlen(buf), buf);

        save_protected(file_buf);

        printf("%s\n",file_buf);



        if (write(sock_fd, buf, 1) < 0) {
            fprintf(stderr,"ERROR: can't write response to client\n");
            break;
        }

    }

    printf("\n--Done session with client--\n");

    free((thread_param_t*)args);
    close(sock_fd);
    
    printf("\n-- child thread call kill--\n");
    pthread_kill(pthread_self(), SIGUSR1);
    printf("\n--child thread call exit--\n");
    pthread_exit(NULL);
    printf("\n--child thread never happend--\n");
}