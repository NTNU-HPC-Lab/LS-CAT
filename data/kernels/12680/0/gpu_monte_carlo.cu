#include "includes.h"
// Source: http://web.mit.edu/pocky/www/cudaworkshop/MonteCarlo/Pi.cu

// Written by Barry Wilkinson, UNC-Charlotte. Pi.cu  December 22, 2010.
//Derived somewhat from code developed by Patrick Rogers, UNC-C
//
//How to run?
//===========
//
//Single precision :
//
//nvcc -O3 pi-curand.cu ; ./a.out <thread_num>
//
//Double precision
//
//nvcc -O3 -D DP pi-curand.cu ; ./a.out <thread_num>


#define TRIALS_PER_THREAD 4096
#define BLOCKS 256
#define THREADS 256


//Help code for switching between Single Precision and Double Precision
#ifdef DP
typedef double Real;
#define PI  3.14159265358979323846  // known value of pi
#else
typedef float Real;
#define PI 3.1415926535  // known value of pi
#endif


/**
A random number generator.
Guidance from from http://stackoverflow.com/a/3067387/1281089
**/
__global__ void gpu_monte_carlo(Real *estimate, curandState *states, int trials) {
unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
int points_in_circle = 0;
Real x, y;

curand_init(1234, tid, 0, &states[tid]);  // 	Initialize CURAND


for(int i = 0; i < trials; i++) {
x = curand_uniform (&states[tid]);
y = curand_uniform (&states[tid]);
points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
}
estimate[tid] = 4.0f * points_in_circle / (Real) trials; // return estimate of pi
}