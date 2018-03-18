#include <mpi.h>
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

int main(int argc, char** argv)
{
    int world_size, world_rank, name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    /* Initialize the MPI environment */
    MPI_Init(NULL, NULL);
    
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);         // Get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);         // Get the rank of the process
    MPI_Get_processor_name(processor_name, &name_len);  // Get the name of the processor
    
    /* Print off a hello world message */
    sleep(world_rank);
    printf("Hello world from processor %s, rank %d out of %d processors\n", 
            processor_name, world_rank, world_size);

    show_info(processor_name);

    /* Finalize the MPI environment */
    MPI_Finalize();
    return 0;
}
