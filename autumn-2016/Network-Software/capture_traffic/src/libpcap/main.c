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
*   flag_mode[7] m- statisitc
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
void handler(u_char *user, const struct pcap_pkthdr *pkthdr, const u_char *buffer)
{
    int size = pkthdr->len;

    printf("\nTime of capture          : %s", ctime(&pkthdr->ts.tv_sec));
    printf("Capture lenght of packet : %d bytes\n", pkthdr->caplen);
    printf("Full lenght of packet    : %d bytes\n", pkthdr->len);

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
    char errbuf[PCAP_ERRBUF_SIZE] = { 0 };
    char filter_exp[256];
    strcpy(filter_exp, "port ");
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
    while ((rc = getopt(argc, argv, "c:o:w:dhtiuas")) != -1) {
        switch (rc)
        {
            case 'c':
                flag_mode[0] = "c"; // count
                count_packet = atoi(optarg);
                printf("[show total packet %d ] ", count_packet);
                break;

            case 'o':
                strcat(filter_exp, optarg);
                printf("[option %s] ", filter_exp);
                break;

            case 'w':
                strcat(word, optarg);
                printf("[word %s] ", word);
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
     * int sock_raw = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
     *
     * Optional:
     * Its important to provide the correct interface name
     * to setsockopt, eth0 in this case and in most cases.
     *
     * setsockopt(sock_raw, SOL_SOCKET, SO_BINDTODEVICE, 
     *            "eth0", strlen("eth0")+1);
     *
     * if (sock_raw < 0) {
     *    //Print the error
     *    return 1;
     * }
     *
     * while (1) {
     *   //Receive a packet
     *   if (recvfrom(sock_raw, buffer) < 0) {
     *       //Print the error
     *   }
     *   callback(args); //process the packet
     * }
     * close(sock_raw);
     */


    char dev[] = /*"enp3s0"*/ "enp0s3";         /* Device to sniff on */
    struct bpf_program fp;         /* The compiled filter expression */
    //char filter_exp[] = "port 21"; /* The filter expression */
    bpf_u_int32 mask;              /* The netmask of our sniffing device */
    bpf_u_int32 net;               /* The IP of our sniffing device */

    if (pcap_lookupnet(dev, &net, &mask, errbuf) == -1) {
         fprintf(stderr, "Can't get netmask for device %s\n", dev);
         net = 0;
         mask = 0;
    }

    //Open the device for sniffing
    //handle = pcap_open_live("eth0", BUFSIZ, 0, -1, errbuf);
    handle = pcap_open_live(dev, BUFSIZ, 0, -1, errbuf);
    if (handle == NULL) {
        fprintf(stderr, "error: Couldn't open device eth0: %s\n", errbuf);
        exit(1);
    }

    if (pcap_compile(handle, &fp, filter_exp, 0, net) == -1) {
        fprintf(stderr, "Couldn't parse filter %s: %s\n", filter_exp, pcap_geterr(handle));
        return(2);
    }

    if (pcap_setfilter(handle, &fp) == -1) {
        fprintf(stderr, "Couldn't install filter %s: %s\n", filter_exp, pcap_geterr(handle));
        return(2);
    }


    //Put the device in sniff loop (100 packets)
    pcap_loop(handle, count_packet, handler, NULL);
    
    // Close the device
    pcap_close(handle);

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

    return 0;
}
