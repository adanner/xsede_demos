#ifndef _CUDA_VECSUM_H
#define _CUDA_VECSUM_H

/* Compute vector sum c=a+scale*b using GPU. a,b,c reside in host memory, 
 * so this function can be called by normal C++ code. The implementation of
 * this function in cuda_vecsum.cu will make CUDA calls. Headers like these 
 * allow the creation of a interface between normal C++ code and CUDA code*/
void addVectors(int* a, int* b, int N, int scale, int* c);

#endif

