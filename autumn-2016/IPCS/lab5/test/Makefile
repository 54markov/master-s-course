all: lsb_r lsb_m hamm hamm1

lsb_r: lsb_r.c
	gcc -g -o lsb_r lsb_r.c

lsb_m: lsb_m.c
	gcc -g -o lsb_m lsb_m.c

hamm: hamming_code.c
	gcc -g -o hamm hamming_code.c

hamm1: hamming_matrix.c
	gcc -g -o hamm1 hamming_matrix.c

    
clean:
	rm -f hamm hamm1 lsb_r lsb_m *.bmp *.tar.gz statistic
