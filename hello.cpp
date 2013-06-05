#include <iostream>

/* run qsub hello in the scripts directory for your cluster
 * no mpi, no cuda. single threaded app */

using namespace std;

int main(){
	cout << "Hello XSEDE!" << endl;
	return 0;
}

