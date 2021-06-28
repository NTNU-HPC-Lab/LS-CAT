#include "includes.h"
__global__ void matrixTranspose(double *a, double *c, int cr, int cc){

int x = blockIdx.x * blockDim.x + threadIdx.x; // col
int y = blockIdx.y * blockDim.y + threadIdx.y; // row


if(x < cc && y < cr){

for(int i = 0; i<cc; i++) {

c[y * cc + x+i] = a[x * cc + y + i];

}
}


}