#include "includes.h"


#define BLKX 32
#define BLKY 32

cudaStream_t gstream;


__global__ void initData(int nbLines, int M, double *h, double *g)
{
long idX = threadIdx.x + blockIdx.x * blockDim.x;

if (idX > nbLines * M)
return;

h[idX] = 0.0L;
g[idX] = 0.0L;
if ( idX >= M +1  && idX  < 2*M-1 ){
h[idX] = 100.0;
g[idX] = 100.0;
}
}