#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void show_info(char *whois)
{
    printf("*** %s ***\n", whois);
    printf("PID  : %u\n", getpid());
    printf("PPID : %u\n", getppid());
    printf("UID  : %u\n", getuid());
    printf("EUID : %u\n", geteuid());
    printf("GID  : %u\n", getgid());
    printf("EGID : %u\n", getegid());
    printf("PG   : %u\n", getpgrp());
    printf("SID  : %u\n", getsid(getpid()));
}

int main(void)
{
    int pid = fork();
    
    if (pid == -1) {
        perror("fork"); 
        exit(1);
    }

    if (pid == 0) {
        sleep(1);
        show_info("Child"); 
    } else {
        sleep(3);
        show_info("Parent");
    }
    
    return (0);
}
