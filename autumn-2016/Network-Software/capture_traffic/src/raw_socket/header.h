#ifndef HEADER_H
#define HEADER_H

#include <pcap.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

#include <unistd.h>

#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/icmp.h>

#include <arpa/inet.h>

#include <linux/if_arp.h>

#include <sys/ioctl.h>		// for SIOCGIFFLAGS, SIOCSIFFLAGS
#include <netinet/in.h>		// for htons()
#include <linux/if_ether.h>	// for ETH_P_ALL
#include <linux/if.h>		// for struct ifreq, IFNAMSIZ

#define COLOR_OFF "\033[0m"

#define COLOR_RED "\033[0;31m"
#define COLOR_CYN "\033[0;36m"
#define COLOR_BLU "\033[0;34m"
#define COLOR_GRN "\033[0;32m"
#define COLOR_RED "\033[0;31m"
#define COLOR_BLC "\033[0;30m"
#define COLOR_WHT "\033[0;37m"
#define COLOR_YEL "\033[0;33m"
#define COLOR_MAG "\033[0;35m"

struct pseudo_header {
    unsigned int   src_ip;
    unsigned int   dest_ip;
    unsigned char  zeroes;
    unsigned char  protocol;
    unsigned short len;
} __attribute__ ((packed));


void print_data(const u_char *data , int size);

void print_tcp_packet(const u_char *data, int size, char *arg1, char *arg2);
void print_udp_packet(const u_char *data, int size, char *arg1, char *arg2);
void print_icmp_packet(const u_char *data, int size, char *arg1, char *arg2);

void print_other_packet(const u_char *data, int size, char *arg1, char *arg2);

void print_ethernet_header(const u_char *data);
void print_ip_header(const u_char *data, int size);
void print_tcp_header(const u_char *data, int size);
void print_arp_header(const u_char *data, int size);

unsigned short csum(unsigned short *data, int len);
uint16_t checksum(uint8_t *buf, uint16_t len);

void compute_tcp_checksum(struct iphdr *pIph, unsigned short *ipPayload); 
unsigned short csum1(unsigned short* vdata, int length, struct iphdr *pIph); 


#endif