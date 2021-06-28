#include "includes.h"

#define NUM_THREADS 	743511 	// length of calculation
#define BLOCK_SIZE 	256	// number of threads per block used in gpu calc
#define EPS		0.00005 // Epsilon for tolerance of diffs between cpu and gpu calculations
#define INCLUDE_MEMTIME false	// Decides whether to include memory transfers to and from gpu in gpu timing
#define PRINTLINES	0	// Number of lines to print in output during validation



int timeval_subtract(  struct timeval* result,
struct timeval* t2,
struct timeval* t1) {
unsigned int resolution = 1000000;
long int diff = (t2->tv_usec + resolution * t2->tv_sec) -
(t1->tv_usec + resolution * t1->tv_sec);
result->tv_sec = diff / resolution;
result->tv_usec = diff % resolution;
return (diff<0);
}

__global__ void calcKernel(float* d_in, float *d_out) {
const unsigned int lid = threadIdx.x;			// local id inside a block
const unsigned int gid = blockIdx.x*blockDim.x + lid; 	// global id
d_out[gid] = pow((d_in[gid] / ( d_in[gid] - 2.3 )),3);	// do computation
}