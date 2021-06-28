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

__global__ void startScanKernel(const int N, const int *v, int *scanv, int *starts){

__shared__ int s_v0[BLOCKSIZE];
__shared__ int s_v1[BLOCKSIZE];

int j = threadIdx.x;
int b = blockIdx.x;
int n = j + b*BLOCKSIZE;

s_v0[j] = (n<N) ?  v[j+b*BLOCKSIZE]: 0;

int offset = 1;
do{
__syncthreads();

s_v1[j] = (j<offset) ? s_v0[j] : (s_v0[j]+s_v0[j-offset]) ;

offset *= 2;

__syncthreads();

s_v0[j] = (j<offset) ? s_v1[j] : (s_v1[j]+s_v1[j-offset]) ;

offset *= 2;
} while(offset<BLOCKSIZE);

if(n<N)
scanv[n+1] = s_v0[j];

if(j==(BLOCKSIZE-1)){
starts[b+1] = s_v0[j];
}

}