#include "includes.h"
__global__ void kernel_offset(int *key, int *idx, int *offset, int size) {

int idxX = threadIdx.x + blockIdx.x*blockDim.x;

if(idxX == 0) {
offset[1] = 0;
}
else if(idxX < size) {
int keyVal = key[idxX];
int keyValPrev = key[idxX-1];
if(keyVal != keyValPrev) {
offset[keyVal+1] = idxX;
}
}
if(idxX == size-1) {
int keyVal = key[idxX];
offset[0] = keyVal+1;
offset[keyVal+2] = size;
}
}