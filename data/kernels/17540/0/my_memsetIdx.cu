#include "includes.h"
__global__ void my_memsetIdx(int* dg_array, int size, int scale){
const int gtid=blockIdx.x*blockDim.x + threadIdx.x;
if(gtid < size){
dg_array[gtid] = gtid*scale;
}
}