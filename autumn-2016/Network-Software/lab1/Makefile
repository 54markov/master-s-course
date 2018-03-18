all: udp_server udp_client

udp_server: server/udp_server.c
	gcc  -g -o udp_server server/udp_server.c server/queue.c

udp_client: client/udp_client.c
	gcc -g -o udp_client client/udp_client.c
	
clean:
	rm -f udp_server udp_client
