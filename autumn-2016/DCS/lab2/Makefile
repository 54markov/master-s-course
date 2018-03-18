all: level0 level1 level2 level3 levelf levelV level main timer

timer:
	gcc -g -o custom_timer custom_timer.c

main:
	gcc -g -o compiler_optimization compiler_optimization.c -lm

level:
	gcc -S -o level.s compiler_optimization.c -lm

level0:
	gcc -S -O0 -o level0.s compiler_optimization.c -lm

level1:
	gcc -S -O1 -o level1.s compiler_optimization.c -lm

level2:
	gcc -S -O2 -o level2.s compiler_optimization.c -lm

level3:
	gcc -S -O3 -o level3.s compiler_optimization.c -lm

levelf:
	gcc -S -Ofast -o levelf.s compiler_optimization.c -lm

levelV:
	gcc -O2 -ftree-vectorize -S -o levelv.s compiler_optimization.c -lm


clean:
	rm -f level* compiler_optimization custom_timer