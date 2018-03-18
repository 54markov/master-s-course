#include "header.h"

void print_ip_header(const u_char *data, int size)
{
    int 		new_sum; // new checksum
    int			old_sum; // old checkum

    unsigned short      iphdrlen;// size ip header
    struct sockaddr_in  source;  // address source ip
    struct sockaddr_in  dest;    // address destination ip
    struct iphdr        *iph;    // ip header

    iph = (struct iphdr*)(data + sizeof(struct ethhdr) );
    iphdrlen = iph->ihl * 4;

    source.sin_addr.s_addr = iph->saddr;
    dest.sin_addr.s_addr = iph->daddr;

    printf(COLOR_CYN);
    printf("\nIP Header\n");
    printf(COLOR_OFF);

    printf("   |-IP Version          : %d\n", (unsigned int)iph->version);
    printf("   |-IP Header Length    : %d DWORDS or %d Bytes\n",
                                (unsigned int)iph->ihl, ((unsigned int)(iph->ihl))*4);
    //printf("   |-Type Of Service     : %d\n", (unsigned int)iph->tos);
    printf("   |-IP Total Lengt      : %d  Bytes(Size of Packet)\n", ntohs(iph->tot_len));
    printf("   |-Identification      : %d\n", ntohs(iph->id));
    printf("   |-TTL                 : %d\n", (unsigned int)iph->ttl);
    printf("   |-Protocol            : %d\n", (unsigned int)iph->protocol);
    printf("   |-Checksum            : %d\n", ntohs(iph->check));
    printf("   |-Source IP           : %s\n", inet_ntoa(source.sin_addr) );
    printf("   |-Destination IP      : %s\n", inet_ntoa(dest.sin_addr) );

    old_sum = ntohs(iph->check); // store old checksum
    iph->check = 0;  // set to zero checksum

    //new_sum = csum((unsigned short*)(data + sizeof(struct ethhdr)), (int)iphdrlen);
	new_sum = checksum((uint8_t*)(data + sizeof(struct ethhdr)), (uint16_t)iphdrlen);

    if (old_sum == new_sum) {
        printf("\n   |-**Recheck Checksum  : %s%d%s (%scorrect%s)\n", 
                COLOR_GRN, 
                new_sum,
                COLOR_OFF,
                COLOR_GRN,
                COLOR_OFF);
    } else {
        printf("\n   |-**Recheck Checksum  : %s%d%s (%serror%s)\n",
                COLOR_RED, 
                new_sum,
                COLOR_OFF,
                COLOR_RED,
                COLOR_OFF);
        //exit(1);
    }
}
