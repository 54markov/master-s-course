#include "header.h"

void print_other_packet(const u_char *data, int size, char *eth, char *dat)
{
    unsigned short 	arphdrlen = sizeof(struct arphdr);

    int 		header_size = sizeof(struct ethhdr) + arphdrlen;

    printf(COLOR_YEL);
    printf("\n****************************OTHER/ARP Packet*******************************\n");  
    printf(COLOR_OFF);

    if (strcmp(eth, "h") == 0) {
	print_ethernet_header(data);
	print_arp_header(data, size);
    }

    if (strcmp(dat, "d") == 0) {
	   printf("\n              ----------DATA Dump----------\n");

       
	   printf("ARP Header\n");
	   print_data((data + sizeof(struct ethhdr)), arphdrlen);

        printf("Data Payload\n");
	   print_data((data + header_size), (size - header_size));
    }

    printf(COLOR_YEL);
    printf("\n***********************************************************\n");
    printf(COLOR_OFF);
}