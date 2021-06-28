/******************************************************************************
COMMENT FROM ALI HELLO
  COMMENNTS HERE

******************************************************************************/

#ifndef GPULib
#define GPULib

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

# define cD2H cudaMemcpyDeviceToHost
# define cH2D cudaMemcpyHostToDevice

struct APopulation{

  dim3 nBlocks;
  dim3 nThreads;
  unsigned long N;

  unsigned long pop_width;       // save this info
  unsigned long pop_height;

  curandState *rand;
  // int *dev_a;
  // int *dev_b;
  // int *dev_c;

  float* red;
  float* green;
  float* blue;

};


APopulation initializePop(unsigned int numBlocks, unsigned int numThreads);
int runIter(APopulation *thePop, unsigned long tick);
void freeGPU(APopulation *thePop);

//__global__ void add(int *a, int *b, int *c);
__global__ void setup_rands(curandState*, unsigned long seed, unsigned long N);
__global__ void randomize(float* array, curandState* rand, unsigned long N);
__global__ void kernel(float *, float *, float *, unsigned long N);


#endif // GPULib
