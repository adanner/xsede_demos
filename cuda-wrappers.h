#ifndef _CUDA_WRAPPERS_H
#define _CUDA_WRAPPERS_H


/* Return number of CUDA devices available on current host,
 * may return 0*/
int getGPUCount();

/* Select and set the current GPU for current thread/process id
 * Assumes the number of processes/threads per host is no greater
 * than the number of GPUs. Returns -1 if there is an error, and a 
 * valid GPU id if no error. The current GPU device will be set to the
 * valid GPU id, so the user does not need to set this after calling the
 * function */
int pickGPU(int rank, int ngpus);

/* Compute vector sum c=a+scale*b using GPU. a,b,c reside in host memory, 
 * so this function can be called by normal C++ code. The implementation of
 * this function in cuda_vecsum.cu will make CUDA calls. Headers like these 
 * allow the creation of a interface between normal C++ code and CUDA code*/
void addVectors(int* a, int* b, int N, int scale, int* c);

#endif

