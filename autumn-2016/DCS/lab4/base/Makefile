all: daemon1 daemon2 process_info

daemon1: daemon1.c
	gcc -g -o daemon1 daemon1.c -std=c11

daemon2: daemon2.c
	gcc -g -o daemon2 daemon2.c -std=c11

process_info: process_info.c
	gcc -g -o process_info process_info.c

clean:
	rm -f daemon2 daemon1 process_info