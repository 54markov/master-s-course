#include "header.h"

void print_tcp_header(const u_char * data, int size)
{
    u_char                  *buf;

    int                     data_gram;
    int                     new_sum; // new checksum
    int	                    old_sum; // old checkum
    int                     header_size;

    unsigned short          iphdrlen;

    struct tcphdr           *tcph;
    struct iphdr            *iph;
    struct pseudo_header    *pseudo_head;

    iph = (struct iphdr*)(data + sizeof(struct ethhdr));

    iphdrlen = iph->ihl * 4;

    tcph = (struct tcphdr*)(data + iphdrlen + sizeof(struct ethhdr));

    printf(COLOR_CYN);
    printf("\nTCP Header\n");
    printf(COLOR_OFF);

    printf("   |-Source Port         : %u\n", ntohs(tcph->source));
    printf("   |-Destination Port    : %u\n", ntohs(tcph->dest));
    printf("   |-Sequence Number     : %u\n", ntohl(tcph->seq));
    printf("   |-Acknowledge Number  : %u\n", ntohl(tcph->ack_seq));
    printf("   |-Header Length       : %d DWORDS or %d BYTES\n" ,
                (unsigned int)tcph->doff, (unsigned int)tcph->doff * 4);
    //printf("   |-Urgent Flag         : %d\n", (unsigned int)tcph->urg);
    //printf("   |-Acknowledgement Flag: %d\n", (unsigned int)tcph->ack);
    //printf("   |-Push Flag           : %d\n", (unsigned int)tcph->psh);
    //printf("   |-Reset Flag          : %d\n", (unsigned int)tcph->rst);
    //printf("   |-Synchronise Flag    : %d\n", (unsigned int)tcph->syn);
    //printf("   |-Finish Flag         : %d\n", (unsigned int)tcph->fin);
    printf("   |-Window              : %d\n", ntohs(tcph->window));
    printf("   |-Checksum            : %d\n", ntohs(tcph->check));
    //printf("   |-Urgent Pointer      : %d\n", tcph->urg_ptr);

    old_sum = ntohs(tcph->check); // store old checksum
    tcph->check = 0;  // set to zero checksum

    data_gram = (ntohs(iph->tot_len)) - iphdrlen;
    
    pseudo_head = malloc(sizeof(struct pseudo_header));
    if (!pseudo_head) {
        fprintf(stderr, "error: malloc(pseudo_tcp_head)\n");
        exit(1);
    }

    // fill pseudo header
    pseudo_head->src_ip     = (unsigned int)    iph->saddr;
    pseudo_head->dest_ip    = (unsigned int)    iph->daddr;
    pseudo_head->zeroes     = (unsigned char)   0;
    pseudo_head->protocol   = (unsigned char)   (IPPROTO_TCP);
    pseudo_head->len        = (unsigned short)  htons(data_gram);

    buf = malloc(sizeof(struct pseudo_header) + data_gram);
    if (buf == NULL) {
       fprintf(stderr, "error: malloc(buf)\n");
      exit(1);
    }

    memcpy(buf, pseudo_head, sizeof(struct pseudo_header));
    memcpy(buf + sizeof(struct pseudo_header), tcph, data_gram);

    new_sum = checksum( ((uint8_t*)(buf)), ((uint16_t)(data_gram+(sizeof(struct pseudo_header)))) );

    //new_sum = csum((unsigned short*)(buf), (int)(sizeof(struct pseudo_header) + data_gram));

    //uint16_t checksum(uint8_t *buf, uint16_t len,uint8_t type)

    //new_sum = (int)checksum((uint8_t*)buf, (uint16_t)(sizeof(struct pseudo_header) + data_gram), 2);

    //void compute_tcp_checksum(struct iphdr *pIph, unsigned short *ipPayload); 
    //compute_tcp_checksum(iph, (unsigned short*)(tcph)); 
    free(buf);
    free(pseudo_head);

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