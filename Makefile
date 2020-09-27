INC="inc"
NVCCFLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall
LFLAGS=-lglut -lGL

all:convolution

convolution: convolution.cu
	$(NVCC) $(NVCCFLAGS) -lineinfo convolution.cu -o convolution $(LFLAGS)

clean:
	rm convolution
