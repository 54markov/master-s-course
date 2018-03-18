#!/bin/sh

export OMP_NUM_THREADS=2
./dgemv

export OMP_NUM_THREADS=4
./dgemv

export OMP_NUM_THREADS=6
./dgemv

export OMP_NUM_THREADS=8
./dgemv

