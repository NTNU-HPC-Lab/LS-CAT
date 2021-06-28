#include "includes.h"


#define BLKX 32
#define BLKY 32

cudaStream_t gstream;


__global__ void gpuWork(double *g, double *h, double *error,  int M, int nbLines){

// This moves thread (0,0) to position (1,1) on the grid
long idX = threadIdx.x + blockIdx.x * blockDim.x +1;
long idY = threadIdx.y + blockIdx.y * blockDim.y +1;
long threadId = threadIdx.y * blockDim.x + threadIdx.x;
long tidX = threadIdx.x + blockIdx.x * blockDim.x;
long tidY = threadIdx.y + blockIdx.y * blockDim.y;

register double temp;
long xSize = M+2;

__shared__ double errors[BLKX*BLKY];

errors[threadId] = 0.0;

if (tidX < M && tidY < nbLines ){
temp = 0.25*(h[(idY-1)*xSize +idX]
+h[((idY+1)*xSize)+idX]
+h[(idY*xSize)+idX-1]
+h[(idY*xSize)+idX+1]);
errors[threadId] = fabs(temp - h[(idY*xSize)+idX]);
g[(idY*xSize)+idX] = temp;
}
else{
return;
}

__syncthreads();


for (unsigned long s = (blockDim.x*blockDim.y)/2; s>0; s=s>>1){
if ( threadId < s ){
errors[threadId] =  fmax(errors[threadId], errors[threadId+s]);
}
__syncthreads();
}


if ( threadId == 0 ){
int id = blockIdx.y * (gridDim.x) + blockIdx.x;
error[id] = errors[0];
}
return;
}