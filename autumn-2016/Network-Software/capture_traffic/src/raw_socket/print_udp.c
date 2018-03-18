#include "header.h"

void print_udp_packet(const u_char *data, int size, char *eth, char *dat)
{
    u_char                  *buf;
    
    int                     new_sum; // new checksum
    int                     old_sum; // old checkum
    int                     header_size; 
    int                     data_gram;

    unsigned short          iphdrlen;

    struct iphdr            *iph;
    struct udphdr           *udph;
    struct pseudo_header    *pseudo_head;

    iph = (struct iphdr*)(data + sizeof(struct ethhdr));
    
    iphdrlen = iph->ihl * 4;
     
    udph = (struct udphdr*)(data + iphdrlen + sizeof(struct ethhdr));
     
    header_size =  sizeof(struct ethhdr) + iphdrlen + sizeof(struct udphdr);

    printf(COLOR_YEL);
    printf("\n*************************UDP Packet*****************************************\n");
    printf(COLOR_OFF);

    if (strcmp(eth, "h") == 0) {
        print_ip_header(data, size);           
        
        printf(COLOR_CYN);
        printf("\nUDP Header\n");
        printf(COLOR_OFF);
        printf("   |-Source Port         : %d\n" , ntohs(udph->source));
        printf("   |-Destination Port    : %d\n" , ntohs(udph->dest));
        printf("   |-UDP Length          : %d\n" , ntohs(udph->len));
        printf("   |-UDP Checksum        : %d\n" , ntohs(udph->check));
    }

    old_sum = ntohs(udph->check); // store old checksum
    udph->check = 0;  // set to zero checksum

    pseudo_head = malloc(sizeof(struct pseudo_header));
    if (pseudo_head == NULL) {
        fprintf(stderr, "error: malloc(pseudo_udp_head)\n");
        exit(1);
    }

    // fill pseudo header
    pseudo_head->src_ip     = (unsigned int)    iph->saddr;
    pseudo_head->dest_ip    = (unsigned int)    iph->daddr;
    pseudo_head->zeroes     = (unsigned char)   0;
    pseudo_head->protocol   = (unsigned char)   iph->protocol;
    pseudo_head->len        = (unsigned short)  udph->len;

    buf = malloc( (sizeof(struct pseudo_header)) + (ntohs(udph->len)) );
    if (buf == NULL) {
        fprintf(stderr, "error: malloc(buf)\n");
        exit(1);
    }

    memcpy( buf, pseudo_head, (sizeof(struct pseudo_header)) );
    memcpy( buf + (sizeof(struct pseudo_header)), udph, (ntohs(udph->len)) );

    data_gram = (sizeof(struct pseudo_header)) + (ntohs(udph->len));

    //new_sum = csum( (unsigned short*)buf, data_gram);
	new_sum = checksum((uint8_t*)(buf), (uint16_t)data_gram);

    free(pseudo_head);
    free(buf);

    if (old_sum == new_sum) {
        printf("\n   |-**Recheck Checksum  : %s%d%s (%scorrect%s)\n", 
                COLOR_GRN, new_sum, COLOR_OFF, COLOR_GRN, COLOR_OFF);
    } else {
        printf("\n   |-**Recheck Checksum  : %s%d%s (%serror%s)\n",
                COLOR_RED, new_sum, COLOR_OFF, COLOR_RED, COLOR_OFF);
        //exit(1);
    }

    if (strcmp(dat, "d") == 0) {
        printf(COLOR_YEL);
        printf("\n              ----------DATA Dump----------\n");
        printf(COLOR_OFF);

        printf(COLOR_CYN);
        printf("\nIP Header\n");
        printf(COLOR_OFF);
        print_data(data, iphdrlen);

        printf(COLOR_CYN);
        printf("\nUDP Header\n");
        printf(COLOR_OFF);
        print_data(data + iphdrlen, sizeof(struct udphdr));

        printf(COLOR_CYN);
        printf("\nData Payload\n");
        printf(COLOR_OFF);

        //Move the pointer ahead and reduce the size of string
        print_data(data + header_size, size - header_size);
    }

    printf(COLOR_YEL);
    printf("\n***************************************************************************\n");                         
    printf(COLOR_OFF);
}
