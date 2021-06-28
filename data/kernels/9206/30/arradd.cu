#include "includes.h"
__global__ void arradd(const int *md, const int *nd, int *pd, int size){
int myid = blockDim.x*blockIdx.x + threadIdx.x;
if(myid < size)
pd[myid] = md[myid] + nd[myid];
}