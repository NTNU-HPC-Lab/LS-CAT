#ifndef FILE_Random_SEEN
#define FILE_Random_SEEN
#include <curand.h>
#include <curand_kernel.h>
/*------------------------------------*/
__global__ void init_random(unsigned long long *seed, curandState  *global_state);
__global__ void random(double *x, curandState *global_state);
__global__ void UniformRandom(double *x, curandState *global_state);
__device__ double Poisson(double xmean, curandState *mystate);
__device__ double Gaussian(double mean, double sigma, curandState *mystate);
/*------------------------------------*/
#endif /* !Random_SEEN */
