all: main.o print_data.o print_tcp.o print_eth_head.o print_ip_head.o print_tcp_head.o print_icmp.o print_udp.o print_other.o checksum.o print_arp_head.o
	gcc -g -o main main.o print_data.o print_tcp.o print_eth_head.o print_ip_head.o print_tcp_head.o print_icmp.o print_udp.o print_other.o checksum.o print_arp_head.o -lpcap
	rm -rf *.o

main.o:
	gcc -c main.c

print_data.o:
	gcc -c print_data.c

print_tcp.o:
	gcc -c print_tcp.c

print_icmp.o:
	gcc -c print_icmp.c

print_udp.o:
	gcc -c print_udp.c

print_eth_head.o:
	gcc -c print_eth_head.c

print_ip_head.o:
	gcc -c print_ip_head.c

print_tcp_head.o:
	gcc -c print_tcp_head.c

print_arp_head.o:
	gcc -c print_arp_head.c

print_other.o:
	gcc -c print_other.c

checksum.o:
	gcc -c checksum.c

clean:
	rm -rf main *.o

