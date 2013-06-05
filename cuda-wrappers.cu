#include "book.h"
#include <cstdlib>
#include "cuda-wrappers.h"

/* Compute vector sum c=a+scale*b on GPU. a,b,c reside on GPU memory.
 * This function must be launched as a CUDA kernel */
__global__ void scalarVecSum( int* a, int* b, int N, int scale, int* c);

/* Return number of CUDA devices available on current host, 
 * returns 0 if error, or no devices found.*/
int getGPUCount(){
	int ans=0;
	if(cudaGetDeviceCount(&ans) != cudaSuccess){
		return 0;
	}
	return ans;
}

/* Select and set the current GPU for current thread/process id
 * Assumes the number of processes/threads per host is no greater
 * than the number of GPUs. Returns -1 if there is an error, and a 
 * valid GPU id if no error. The current GPU device will be set to the
 * valid GPU id, so the user does not need to set this after calling the
 * function */
int pickGPU(int rank, int ngpus){
	int gpuID = rank%ngpus; //any better way?
	if(cudaSetDevice(gpuID) != cudaSuccess){
		return -1;
	}
	return gpuID;
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

/* Compute vector sum c=a+scale*b on GPU. a,b,c reside on GPU memory.
 * This function must be launched as a CUDA kernel */
__global__ void scalarVecSum( int* a, int* b, int N, int scale, int* c) {
	// this thread handles the data at its thread id
	int tid = blockIdx.x;    
	if (tid < N){
		c[tid] = a[tid] + scale*b[tid];
	}
}


