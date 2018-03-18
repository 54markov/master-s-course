all: sqrt_sse sqrt_avx

sqrt_sse: sqrt_sse.c
	gcc -g -o sqrt_sse sqrt_sse.c -std=c99 -msse3 -lm

sqrt_avx: sqrt_avx.c
	gcc -g -o sqrt_avx sqrt_avx.c -std=c99 -mavx -lm

clean:
	rm -rf sqrt_sse sqrt_avx