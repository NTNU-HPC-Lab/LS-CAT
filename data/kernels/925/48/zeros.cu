#include "includes.h"
__global__ void zeros(double *field, int n){
int xid = blockDim.x*blockIdx.x + threadIdx.x;

if (xid < n){
field[xid] = 0;
}

}