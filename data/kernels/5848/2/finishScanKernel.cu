#include "includes.h"
// risky
#define dfloat double

#define p_eps 1e-6

#define p_Nsamples 1

// ratio of importance in sampling primary ray versus random rays
#define p_primaryWeight 2.f

#define p_intersectDelta 0.1f

#define p_shadowDelta 0.15f
#define p_projectDelta 1e-2

#define p_maxLevel 5
#define p_maxNrays (2<<p_maxLevel)
#define p_apertureRadius 20.f
#define NRANDOM 10000

cudaEvent_t startTimer, endTimer;

__global__ void finishScanKernel(const int N, int *scanv, int *starts){

int j = threadIdx.x;
int b = blockIdx.x;

int n=j+b*BLOCKSIZE;

if(n<N){
int start = starts[b];

scanv[n+1] += start;
}
}