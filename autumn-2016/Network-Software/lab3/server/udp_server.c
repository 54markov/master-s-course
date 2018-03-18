#include "multi_server.h"

#define LOST_PACKET 3

void doprocessing_udp(int udp_fd)
{
    static int call_counter = 0;

    char buffer[256] = { 0 };
    struct sockaddr_in  client_addr;
    int client_len = sizeof(client_addr);

    // accept new connection from client
    int r_bytes = recvfrom(udp_fd , buffer, 256, 0,(struct sockaddr *)&client_addr, &client_len);
    if (r_bytes == -1) {
        fprintf(stderr, "ERROR: recvfrom()");
    }

    char *hostaddrp = inet_ntoa(client_addr.sin_addr);
    if (!hostaddrp) {
        fprintf(stderr, "ERROR on inet_ntoa\n");
        return;
    }

    if ((call_counter % LOST_PACKET) != 0) {
        fprintf(stderr, "NOTE: simulate network packet loss\n");
        call_counter++;
        return;
    }

    printf("received vid udp: (%s) port %d %d bytes: %s\n", hostaddrp, client_addr.sin_port, (int)strlen(buffer), buffer);

    if (sendto(udp_fd, buffer, 1, 0, (struct sockaddr *)&client_addr, sizeof(client_addr)) == -1) {
        fprintf(stderr, "ERROR: sendto()");
        return;
    }

    call_counter++;
}