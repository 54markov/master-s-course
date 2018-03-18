all: sdot_sse sdot_avx

sdot_sse: sdot_sse.c
	gcc -g -o sdot_sse sdot_sse.c -std=c99 -msse3

sdot_avx: sdot_avx.c
	gcc -g -o sdot_avx sdot_avx.c -std=c99 -mavx

clean:
	rm -rf sdot_sse sdot_avx
