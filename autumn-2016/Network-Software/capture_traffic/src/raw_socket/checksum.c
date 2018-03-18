#include "header.h"

uint16_t checksum(uint8_t *buf, uint16_t len)
{
        uint32_t sum = 0;

        // build the sum of 16bit words
        while(len >1) {
            sum += 0xFFFF & (*buf<<8|*(buf+1));
            buf+=2;
            len-=2;
        }
        // if there is a byte left then add it (padded with zero)
        if (len){
            sum += (0xFF & *buf)<<8;
        }
        // now calculate the sum over the bytes in the sum
        // until the result is only 16bit long
        while (sum>>16) {
            sum = (sum & 0xFFFF)+(sum >> 16);
        }
        // build 1's complement:
        return( (uint16_t) sum ^ 0xFFFF);
}


unsigned short csum(unsigned short* vdata, int length) 
{
    // Cast the data pointer to one that can be indexed.
    char* data=(char*)vdata;

    // Initialise the accumulator.
    uint32_t acc=0xffff;

    // Handle complete 16-bit blocks.
    size_t i=0;
    for (i = 0; i+1 < length; i += 2) {
        uint16_t word;
        memcpy(&word,data+i,2);
        acc+=ntohs(word);
        if (acc>0xffff) {
            acc-=0xffff;
        }
    }

    // Handle any partial block at the end of the data.
    if (length&1) {
        uint16_t word=0;
        memcpy(&word,data+length-1,1);
        acc+=ntohs(word);
        if (acc>0xffff) {
            acc-=0xffff;
        }
    }

    // Return the checksum in network byte order.
    return ~acc;
}

// set tcp checksum: given IP header and tcp segment 
void compute_tcp_checksum(struct iphdr *pIph, unsigned short *ipPayload) 
{
    register unsigned long sum = 0;

    unsigned short tcpLen = ntohs(pIph->tot_len) - (pIph->ihl<<2);

    struct tcphdr *tcphdrp = (struct tcphdr*)(ipPayload);

    //add the pseudo header 

    //the source ip
    sum += (pIph->saddr>>16)&0xFFFF;
    sum += (pIph->saddr)&0xFFFF;

    //the dest ip
    sum += (pIph->daddr>>16)&0xFFFF;
    sum += (pIph->daddr)&0xFFFF;

    //protocol and reserved: 6
    sum += htons(IPPROTO_TCP);

    //the length
    sum += htons(tcpLen);
 
    //add the IP payload
    //initialize checksum to 0
    tcphdrp->check = 0;
    while (tcpLen > 1) {
        sum += htons(*ipPayload++);
        tcpLen -= 2;
    }
    //if any bytes left, pad the bytes and add
    if(tcpLen > 0) {
        //printf("+++++++++++padding, %d\n", tcpLen);
        sum += ((*ipPayload)&htons(0xFF00));
    }
      //Fold 32-bit sum to 16 bits: add carrier to result
      while (sum>>16) {
          sum = (sum & 0xffff) + (sum >> 16);
      }
      sum = ~sum;
    //set computation result
    tcphdrp->check = (unsigned short)sum;
    printf("--->%d", htons(tcphdrp->check));
}

unsigned short csum1(unsigned short* vdata, int length, struct iphdr *pIph) 
{
    // Cast the data pointer to one that can be indexed.
    char* data=(char*)vdata;

    int acc = 0;

    unsigned short tcpLen = ntohs(pIph->tot_len) - (pIph->ihl<<2);

    //add the pseudo header 

    //the source ip
    acc += pIph->saddr;
    acc += pIph->saddr;

    //the dest ip
    acc += pIph->daddr;
    acc += pIph->daddr;

    //protocol and reserved: 6
    acc += htons(IPPROTO_TCP);

    //the length
    acc += htons(tcpLen); 

    // Handle complete 16-bit blocks.
    size_t i = 0;
    for (i = 0; i+1 < length; i += 2) {
        uint16_t word;
        memcpy(&word,data+i,2);
        acc+=ntohs(word);
        if (acc>0xffff) {
            acc-=0xffff;
        }
    }

    // Handle any partial block at the end of the data.
    if (length&1) {
        uint16_t word=0;
        memcpy(&word,data+length-1,1);
        acc+=ntohs(word);
        if (acc>0xffff) {
            acc-=0xffff;
        }
    }

    // Return the checksum in network byte order.
    return ~acc;
}