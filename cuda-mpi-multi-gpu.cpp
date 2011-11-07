#include "cuda-wrappers.h"
#include "mpi.h"
#include <cstdlib>
#include <cstdio>
#include <iostream>

/* creates two arrays a and b of n int (specified on the command line
 * and computes the vector sum c=a+k*b, where k is a scalar. prints 
 * the results and exits */
int main(int argc, char** argv){
  
	if (argc != 2){
		printf("Usage: %s <size>\n", argv[0]);
		return 1;
	}

	int n = atoi(argv[1]);
	int rank, size;

	MPI::Init();
	rank = MPI::COMM_WORLD.Get_rank();
	size = MPI::COMM_WORLD.Get_size();
  std::cout << "Hello, world! I am " << rank << " of " << size << std::endl;
	
	int* a = new int[n];
	int* b = new int[n];
	int* c = new int[n];

	for(int i=0; i<n; i++){
		a[i]=i;
		b[i]=i*i;
		c[i]=0;
	}

	//Get number of GPUs per physical host
	int ngpus = getGPUCount();
 
	std::cout << "I see " << ngpus << " CUDA devices on this host " 
		<< std::endl;

	if(ngpus == 0){
		printf("No GPUs found. Exiting");
		return 2;
	}

	int myGPU = pickGPU(rank,ngpus);
	std::cout << "Using gpu ID " << myGPU << std::endl;

  if(myGPU == -1){
    printf("Unable to select GPU. Ensure MPI processes"
	  		" per host is no greater than number of GPUs");
		return 3;
  }
	
	//Call the function that hides all the CUDA details
  std::cout << "Running CUDA kernel...." << std::endl;
	addVectors(a,b,n,rank,c);

	// display the results
	for (int i=0; i<n; i++) {
		printf( "%d + %d*%d = %d\n", a[i], rank, b[i], c[i] );
	}

	delete [] a;
	delete [] b;
	delete [] c;
	
	MPI::Finalize();
	
	return 0;
}
