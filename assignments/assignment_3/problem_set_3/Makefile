NVCC=nvcc 
MPIXX=mpicxx

CUDA_INCLUDEPATH=/usr/local/cuda/include

NVCC_OPTS=-arch=sm_30 
GCC_OPTS=-std=c++11 -g -O3 -Wall 
CUDA_LD_FLAGS=-L -lcuda -lcudart

final: main.o pde.o jmpi
	g++ -o pde main.o jacobi_kernels.o $(CUDA_LD_FLAGS)

main.o:main.cpp jacobi_kernel.h utils.h 
	g++ -c $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) main.cpp

pde.o: jacobi_kernels.cu jacobi_kernel.h  utils.h
	$(NVCC) -c jacobi_kernels.cu $(NVCC_OPTS)

jmpi: jacobi_mpi.cpp
	$(MPIXX) -o pde_mpi jacobi_mpi.cpp

clean:
	rm *.o pde pde_mpi
