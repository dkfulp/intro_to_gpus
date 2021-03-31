NVCC=nvcc 

CUDA_INCLUDEPATH=/usr/local/cuda/include

NVCC_OPTS=-arch=sm_30 
GCC_OPTS=-std=c++11 -g -O3 -Wall 
CUDA_LD_FLAGS=-L -lcuda -lcudart

final: main.o pde.o
	g++ -o pde main.o jacobi_kernels.o $(CUDA_LD_FLAGS)

main.o:main.cpp jacobi_kernel.h utils.h 
	g++ -c $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) main.cpp

pde.o: jacobi_kernels.cu jacobi_kernel.h  utils.h
	$(NVCC) -c jacobi_kernels.cu $(NVCC_OPTS)

clean:
	rm *.o pde
