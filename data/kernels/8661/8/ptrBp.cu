#include "includes.h"
__device__ void trace_subm(int j, int k, int *daG, int *dbG, double *AB, double *A){
int l;
for(l=0; l<(*dbG); l++){
*(A+j*(*daG)+k) += *(AB+j*(*dbG)+l+k*(*dbG)+l);
}
}
__global__ void ptrBp(int *daG, int *dbG, double *ABg, double *Ag) {
int k = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
trace_subm(j, k, daG, dbG, ABg, Ag);
}