#include "header.h"

void print_tcp_packet(const u_char *data, int size, char *eth, char *dat)
{
    int         header_size;
    unsigned short  iphdrlen;
    struct tcphdr   *tcph;
    struct iphdr    *iph;

    iph = (struct iphdr*)(data + sizeof(struct ethhdr) );
    
    iphdrlen = iph->ihl * 4;
     
    tcph = (struct tcphdr*)(data + iphdrlen + sizeof(struct ethhdr));
             
    header_size = sizeof(struct ethhdr) + iphdrlen + tcph->doff * 4;

    printf(COLOR_YEL);
    printf("\n****************************TCP Packet*************************************\n");  
    printf(COLOR_OFF);

    if (strcmp(eth, "h") == 0) {
       print_ethernet_header(data);
       print_ip_header(data, size);
       print_tcp_header(data, size);
    }

    if (strcmp(dat, "d") == 0) {
        printf(COLOR_YEL);
        printf("\n              ----------DATA Dump----------\n");
        printf(COLOR_OFF);

        printf(COLOR_CYN);
        printf("IP Header\n");
        printf(COLOR_OFF);
        print_data((data + sizeof(struct ethhdr)), iphdrlen);

        printf(COLOR_CYN);
        printf("\nTCP Header\n");
        printf(COLOR_OFF);
        print_data(data + sizeof(struct ethhdr) + iphdrlen, tcph->doff * 4);

        printf(COLOR_CYN);
        printf("\nData Payload\n");
        printf(COLOR_OFF);
        print_data((data + header_size), (size - header_size));
    }

    printf(COLOR_YEL);
    printf("\n***************************************************************************\n");
    printf(COLOR_OFF);    
}