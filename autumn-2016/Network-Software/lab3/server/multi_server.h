#ifndef _MULTI_SERVER_H_
#define _MULTI_SERVER_H_

#include <stdio.h>
#include <stdlib.h>

#include <netdb.h>
#include <netinet/in.h>

#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <arpa/inet.h>

#include <string.h>

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <pthread.h>

#include <unistd.h>
#include <sys/syscall.h>

#define gettid() syscall(SYS_gettid)

#define tkill(tid, sig) syscall(SYS_tkill, tid, sig)

#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct
{
    int sock_fd;
    struct sockaddr_in cli_addr;
} thread_param_t;

pthread_mutex_t file_mutex;


void save_protected(char *buf);
void *doprocessing(void *args);

void doprocessing_udp(int udp_fd);

#endif /* _MULTI_SERVER_H_ */