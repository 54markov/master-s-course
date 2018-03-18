#include "header.h"

void print_ethernet_header(const u_char *data)
{
	int 		i;
	struct ethhdr 	*eth;

	eth = (struct ethhdr*)data;

	printf(COLOR_CYN);
	printf("\nEthernet Header\n");
	printf(COLOR_OFF);

	printf("   |-Destination Address ");

	for (i = 0; i < 6; i++) {
	    printf(": %.2X", eth->h_dest[i]); 
	}

	printf("\n   |-Source Address      ");
	for (i = 0; i < 6; i++) {
	    printf(": %.2X", eth->h_source[i]); 
	}

	if((ntohs(eth->h_proto)) == ETH_P_LOOP) {
	    printf("\n   |-Protocol            : 0x%02X (ETH_LOOP)\n", 
						ntohs(eth->h_proto));
	}

	else if((ntohs(eth->h_proto)) == ETH_P_IP) {
	    printf("\n   |-Protocol            : 0x%02X (IP)\n", 
					    ntohs(eth->h_proto));
	}

	else if((ntohs(eth->h_proto)) == ETH_P_ARP ) {
	    printf("\n   |-Protocol            : 0x%02X (ARP)\n", 
					    ntohs(eth->h_proto));
	} else {
	    printf("\n   |-Protocol            : 0x%02X (OTHER)\n", 
					    ntohs(eth->h_proto));
	}
	//printf("\n   |-Protocol            :%u\n", (unsigned short)eth->h_proto);
}