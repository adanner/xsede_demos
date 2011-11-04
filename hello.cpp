#include <iostream>

/* run with qsub longhorn-hello in the jobscripts directory
 * no mpi, no cuda. single threaded app */

using namespace std;

int main(){
	cout << "Hello Teragrid!" << endl;
	return 0;
}

