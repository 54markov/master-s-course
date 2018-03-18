#include "header.h"

struct arp_head {
    u_char arp_sha[ETH_ALEN]; // sender hardware address
    u_char arp_sip[4];        // sender ip address
    u_char arp_tha[ETH_ALEN]; // target hardware address
    u_char arp_tip[4];        // target ip address
} __attribute__ ((packed));

void print_arp_header(const u_char *data, int size) 
{
    int i;
    int header_size;
    
    struct arphdr *arph;
    struct arp_head *a_hd;

    unsigned short arphdrlen;

    arph = (struct arphdr*)(data + sizeof(struct ethhdr));

    arphdrlen = sizeof(struct arphdr);

    header_size = sizeof(struct ethhdr) + arphdrlen;

    a_hd = (struct arp_head*)(data + header_size);

    printf("\nARP HEADER\n");
    printf("   |-Hardware address    :Ethernet 0x%.2X\n", ntohs(arph->ar_hrd));
    printf("   |-Protocol address    :IP 0x%.2x\n", ntohs(arph->ar_pro));

    printf("   |-Hardware size       :%X\n", arph->ar_hln);
    printf("   |-Protocol size       :%X\n", arph->ar_pln);
    printf("   |-Command             :%X\n", ntohs(arph->ar_op));

    printf("\nADDITIONAL ARP HEADER\n");
    
    printf("   |-MAC sender ");
    for (i = 0; i < 6; i++) {
        printf(":%.2X", a_hd->arp_sha[i]);
    }

    printf("\n   |-IP sender  :");
    for (i = 0; i < 4; i++) {
        if (i == 3) {
            printf("%d", a_hd->arp_sip[i]);
        } else {
            printf("%d.", a_hd->arp_sip[i]);
        }
    }
    printf("\n   |-MAC target ");


    for (i = 0; i < 6; i++) {
        printf(":%.2X", a_hd->arp_tha[i]);
    }

    printf("\n   |-IP target  :");
    for (i = 0; i < 4; i++) {
        if (i == 3) {
           printf("%d", a_hd->arp_tip[i]);
        } else {
           printf("%d.", a_hd->arp_tip[i]);
        }
    }
}