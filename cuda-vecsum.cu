#include "book.h"
#include <cstdlib>
#include "cuda-vecsum.h"

/* Compute vector sum c=a+scale*b on GPU. a,b,c reside on GPU memory.
 * This function must be launched as a CUDA kernel */
__global__ void scalarVecSum( int* a, int* b, int N, int scale, int* c) {
    int tid = blockIdx.x;    // this thread handles the data at its thread id
    if (tid < N){
        c[tid] = a[tid] + scale*b[tid];
		}
}

/* Compute vector sum c=a+scale*b using GPU. a,b,c reside in host memory, 
 * so this function can be called by normal C++ code*/
void addVectors(int* a, int* b, int N, int scale, int* c){
    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, N * sizeof(int) ) );

		// copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );

    scalarVecSum<<<N,1>>>( dev_a, dev_b, N, scale, dev_c );

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( c, dev_c, N * sizeof(int),
                              cudaMemcpyDeviceToHost ) );


    // free the memory allocated on the GPU
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_c ) );
    return;
}

