/*
*   Simple packet sniffer using libpcap library by Markov V.A.
*
*   make clean && make
*   sudo ./main [flags]
*
*   flags:
*   [-i] - show icmp
*   [-u] - show udp 
*   [-t] - show tcp
*   [-a] - show arp
*
*   [-h] - show header (use with -i/-t/-u/-a)
*   [-d] - show data (use with -i/-t/-u/-a)
*   [-c] - count packets
*   [-s] - show statistic
*/

#include "header.h"

/*
*   flag_mode[0] - print header
*   flag_mode[1] - count packets
*   flag_mode[2] - print data
*   flag_mode[3] - icmp
*   flag_mode[4] - tcp
*   flag_mode[5] - udp
*   flag_mode[6] - arp
*   flag_mode[7] - statisitc
*/
char *flag_mode[7] = {"q","q","q","q","q","q","q"};

/*
*   packet_cnt[0] - tcp 
*   packet_cnt[1] - udp
*   packet_cnt[2] - icmp
*   packet_cnt[3] - igmp
*   packet_cnt[4] - other
*   packet_cnt[5] - total
*   packet_cnt[6] - arp
*/
int packet_cnt[7] = { 0 };

// Process the sniffed packet (callback method)
void handler(const u_char *buffer, int size)
{
/*
    int size = pkthdr->len;

    printf("\nTime of capture          : %s", ctime(&pkthdr->ts.tv_sec));
    printf("Capture lenght of packet : %d bytes\n", pkthdr->caplen);
    printf("Full lenght of packet    : %d bytes\n", pkthdr->len);
*/
    //Get the IP Header part of this packet , excluding the ethernet header
    struct iphdr *iph = (struct iphdr*)(buffer + sizeof(struct ethhdr));

    ++packet_cnt[5]; // increment total packet

    //Check the Protocol and do accordingly...
    switch (iph->protocol) 
    {
        /* ICMP Protocol */
        case 1:
        {
            ++packet_cnt[2]; // increment icmp pakset
            if (flag_mode[3] == "i") {
                print_icmp_packet(buffer, size, flag_mode[2], flag_mode[1]);
            }
            //exit(1);
            break;
        }
        /* IGMP Protocol */
        case 2:
        {
            ++packet_cnt[3]; // increment igmp packet
            break;
        }
        /* IP Protocol */
        case 4:
        {
            break;
        }
        /* TCP Protocol */
        case 6:
        {
            ++packet_cnt[0]; // increment tcp packet
            if (flag_mode[4] == "t") {
                print_tcp_packet(buffer, size, flag_mode[2], flag_mode[1]);
            }
            break;
        }
        /* UDP Protocol */
        case 17:
        {
            ++packet_cnt[1]; // increment udp packet
            if (flag_mode[5] == "u") {
                print_udp_packet(buffer, size, flag_mode[2], flag_mode[1]);
            }
            break;
        }
        /* Some Other Protocol like ARP etc. */
        default:
        {
            if (flag_mode[6] == "a") {
                print_other_packet(buffer, size, flag_mode[2], flag_mode[1]);
            }
            ++packet_cnt[4]; // increment other/arp packet
            //exit(1);
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    int rc = 0;
    struct sockaddr saddr;
    char errbuf[PCAP_ERRBUF_SIZE] = { 0 };
    pcap_t *handle = NULL;

    /* Default packet */
    int count_packet = 100;

    /* If no args */
    if (argc < 2) { 
        fprintf(stderr, COLOR_RED);
        fprintf(stderr, "error: no argc\n");
        fprintf(stderr, "usage: %s\n", argv[0]);
        fprintf(stderr, "-c (count packets)\n");
        fprintf(stderr, "-d (print data)\n");
        fprintf(stderr, "-i (print icmp)\n");
        fprintf(stderr, "-a (print arp)\n");
        fprintf(stderr, "-t (print tcp)\n");
        fprintf(stderr, "-u (print udp)\n");
        fprintf(stderr, "-h (print header)\n");
        fprintf(stderr, "-s (show statistic)\n");
        fprintf(stderr, COLOR_OFF);
        exit(1);
    }
    
    // Set up the working mode, set up flags
    while ((rc = getopt(argc, argv, "c:dhtiuas")) != -1) {
        switch (rc)
        {
            case 'c':
                flag_mode[0] = "c"; // count
                count_packet = atoi(optarg);
                printf("[show total packet %d ] ", count_packet);
                break;

            case 'd':
                flag_mode[1] = "d"; // data
                printf("[show data] ");
                break;

            case 'h':
                flag_mode[2] = "h"; // header
                printf("[show header] ");
                break;

            case 't':
                flag_mode[4] = "t"; // tcp
                printf("[show tcp] ");
                break;

            case 'i':
                flag_mode[3] = "i"; // icmp
                printf("[show icmp] ");
                break;

            case 'u':
                flag_mode[5] = "u"; // udp
                printf("[show udp] ");
                break;

            case 'a':
                flag_mode[6] = "a"; // arp
                printf("[show arp] ");
                break;

            case 's':
                flag_mode[7] = "s"; // arp
                printf("[show statistic] ");
                break;

            case '?':
                fprintf(stderr, COLOR_BLU);
                fprintf(stderr, "help\n");
                fprintf(stderr, "usage: %s\n", argv[0]);
                fprintf(stderr, "-c (count packets)\n");
                fprintf(stderr, "-s (show statistic)\n");
                fprintf(stderr, "-i (print icmp)\n");
                fprintf(stderr, "-t (print tcp)\n");
                fprintf(stderr, "-a (print arp)\n");
                fprintf(stderr, "-u (print udp)\n");
                fprintf(stderr, "-d (print data)\n");
                fprintf(stderr, "-h (print header)\n");
                fprintf(stderr, COLOR_OFF);
                exit(1);
                break;
        }
    }

    /*
     * Example how to do it without pcap library:
     * 1. Sniff both incoming and outgoing traffic.
     * 2. Sniff ALL ETHERNET FRAMES, which includes all kinds of
     * IP packets and even more if there are any.
     * 3. Provides the Ethernet headers too, which contain the mac addresses.
     *
     */

    struct ifreq ifopts;    /* set promiscuous mode */
    char ifName[] = "eth0";

    strncpy(ifopts.ifr_name, ifName, IFNAMSIZ-1);

    int sock_raw = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));   

    if (sock_raw < 0) {
        fprintf(stderr, "Can't create raw socket\n");
        return -1;
    }

    // enable 'promiscuous mode' for the selected socket interface
    ioctl(sock_raw, SIOCGIFFLAGS, &ifopts);
    ifopts.ifr_flags |= IFF_PROMISC; // enable 'promiscuous' mode
    ioctl(sock_raw, SIOCSIFFLAGS, &ifopts);

    setsockopt(sock_raw, SOL_SOCKET, SO_BINDTODEVICE, "eth0", strlen("eth0")+1);

    for (int i = 0; i < count_packet; i++) {
        int saddr_size = sizeof saddr;
        char buffer[65536];

        //Receive a packet
        int data_size = recvfrom(sock_raw, buffer, 65536, 0 , &saddr, (socklen_t*)&saddr_size);
        if (data_size < 0) {
            fprintf(stderr, "Recvfrom error, failed to get packets\n");
            return -1;
        }
        //Now process the packet
        handler(buffer, data_size);
    }
    close(sock_raw);

    // Display statistic
    if (flag_mode[7] == "s") {
        char buf_stat[256] = { 0 };
        
        printf(COLOR_MAG);
        printf("+-------------------------------------------------------------------------+");
        printf("\n|-Capturing packets:                                                      |\n");
        
        int len = sprintf(buf_stat, "|-TCP:%d, UDP:%d, ICMP:%d, IGMP:%d, Others: %d, Total: %d",    
            packet_cnt[0],packet_cnt[1],packet_cnt[2],packet_cnt[3],packet_cnt[4],packet_cnt[5]);
        printf("%s", buf_stat);

        for (int i = len; i < 74; i++) {
            printf(" ");
        }
        
        printf("|");
        printf("\n+-------------------------------------------------------------------------+\n");
        printf(COLOR_OFF);  
    }

    return(0);
}
