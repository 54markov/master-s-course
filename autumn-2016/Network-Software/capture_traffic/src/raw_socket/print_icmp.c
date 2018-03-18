#include "header.h"

void print_icmp_packet(const u_char *data, int size, char *eth, char *dat)
{
    int new_sum; // new checksum
    int old_sum; // old checkum
    
    struct iphdr *iph;
    struct icmphdr *icmph;

    unsigned short iphdrlen;

    iph = (struct iphdr*)(data + sizeof(struct ethhdr));

    iphdrlen = iph->ihl * 4;

    icmph = (struct icmphdr*)(data + iphdrlen + sizeof(struct ethhdr));

    printf(COLOR_YEL);
    printf("\n\n***********************ICMP Packet***************************************\n");
    printf(COLOR_OFF);

    if (strcmp(eth, "h") == 0) { 
        print_ethernet_header(data);
        print_ip_header(data, size);

        printf(COLOR_CYN);
        printf("\nICMP Header\n");
        printf(COLOR_OFF);
        printf("   |-Type                : %d",(unsigned int)(icmph->type));

        switch(icmph->type)
        {
            case 0:
                printf("  (ICMP Echo Request)\n");
                break;
            case 8:
                printf("  (ICMP Echo Reply)\n");
                break;
            case 11:
                printf("  (TTL Expired)\n");
                break;
            case 12:
                printf("  (IP Header Error)\n");
                break;
            default:
                printf("  (Other)\n");
                break;
        }

        //printf("   |-Code                : %d\n", (unsigned int)(icmph->code));
        printf("   |-Checksum            : %d\n", ntohs(icmph->checksum));
        //printf("   |-ID       : %d\n", ntohs(icmph->un->echo->id));
        //printf("   |-Sequence : %d\n\n", ntohs(icmph->sequence));

        old_sum = ntohs(icmph->checksum); // store old checksum
        icmph->checksum = 0;  // set to zero checksum

        new_sum = csum((unsigned short*)(data + sizeof(struct ethhdr) + iphdrlen), 
                (size - (sizeof(struct ethhdr) + iphdrlen)));

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

    if (strcmp(dat, "d") == 0) {
        printf(COLOR_YEL);
        printf("\n              ----------DATA Dump----------\n");
        printf(COLOR_OFF);

        printf(COLOR_CYN);
        printf("IP Header\n");
        printf(COLOR_OFF);

        print_data(data + sizeof(struct ethhdr), iphdrlen);

        printf(COLOR_CYN);
        printf("\nICMP Header\n");
        printf(COLOR_OFF);

        print_data(data + sizeof(struct ethhdr) + iphdrlen, sizeof(struct icmphdr));

        printf(COLOR_CYN);
        printf("\nData Payload\n");
        printf(COLOR_OFF);
        //Move the pointer ahead and reduce the size of string
        print_data((data + sizeof(struct ethhdr) + iphdrlen + sizeof(struct icmphdr)),
        (size - (sizeof(struct ethhdr) + iphdrlen + sizeof(struct icmphdr))));
    }
    printf(COLOR_YEL);
    printf("\n***************************************************************************\n");                         
    printf(COLOR_OFF);
}