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

void handler(int sig)
{
   pid_t pid;
   pid = wait(NULL);
   printf("Pid %d exited.\n", pid);
}

void doprocessing(int sock_fd)
{
    int bytesReceived;
    char recvBuff[256] = { 0 };
    char buf[256] = { 0 };

    if (write(sock_fd, "start", 5) < 0) {
        fprintf(stderr, "ERROR writing to socket");
        close(sock_fd);
        return;
    }
 
    if (read(sock_fd, buf, 256) < 0) {
        fprintf(stderr, "ERROR: bytes received <%d> %s\n", bytesReceived, buf);
        close(sock_fd);
        return;
    }

    printf("Bytes received <%d> %s\n", bytesReceived, buf);    

    /* Create file where data will be stored */
    FILE *fp = fopen(buf, "ab"); 
    if (!fp) {
        fprintf(stderr, "ERROR: opening file\n");
        close(sock_fd);
        return ;
    }

    /* Receive data in chunks of 256 bytes */
    while((bytesReceived = read(sock_fd, recvBuff, 256)) > 0) {
        printf("Bytes received <%d>\n", bytesReceived);    
        fwrite(recvBuff, 1, bytesReceived, fp);
    }

    if(bytesReceived < 0) {
        fprintf(stderr, "ERROR: read\n");
    }

    fclose(fp);
    close(sock_fd);
    printf("Child end\n");
    return;
}

int main( int argc, char *argv[])
{
    int sockfd, newsockfd, clilen;
    char buffer[256];
    struct sockaddr_in serv_addr, cli_addr;
    int n, pid, lenght;
   
    /* First call to socket() function */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
   
    if (sockfd < 0) {
      perror("ERROR opening socket");
      exit(1);
   }
   
    /* Initialize socket structure */
    bzero((char *) &serv_addr, sizeof(serv_addr));
   
    serv_addr.sin_family      = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port        = 0;
   
    /* Now bind the host address using bind() call.*/
    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        fprintf(stderr, "ERROR: binding\n");
        return 1;
    }

    lenght = sizeof(serv_addr);
    if (getsockname(sockfd, (struct sockaddr *) &serv_addr, &lenght)) {
        fprintf(stderr, "ERROR: getsockname is failed\n");
        return 1;
    }

    printf("Server: port number <%d>\n", ntohs(serv_addr.sin_port));
   
    /* 
     * Now start listening for the clients, here
     * process will go in sleep mode and will wait
     * for the incoming connection
     */

    listen(sockfd, 10);
    clilen = sizeof(cli_addr);

    signal(SIGCHLD, handler);
   
    while (1) {
        newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);

        if (newsockfd < 0) {
            perror("ERROR on accept");
            exit(1);
        }
      
        /* Create child process */
        pid = fork();
		
        if (pid < 0) {
            perror("ERROR on fork");
            exit(1);
        }
      
        if (pid == 0) {
            /* This is the client process */
            close(sockfd);
            doprocessing(newsockfd);
            exit(0);
        } else {
            close(newsockfd);
        }
    } /* end of while */

    return 0;
}