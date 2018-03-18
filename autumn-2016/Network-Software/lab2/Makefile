all: client server

server: server/tcp_server.c
	gcc -g -o server/tcp_server server/tcp_server.c

client: client/tcp_client.c
	gcc -g -o client/tcp_client client/tcp_client.c

clean:
	rm -f client/tcp_client server/tcp_server