#include "includes.h"
__global__ void initSquare(float* a, float* x, float totalX, int n, int ghosts){
int i = threadIdx.x + blockDim.x*blockIdx.x;
for(int j = 0; blockDim.x*j + i < n; j++){
int index = j*blockDim.x+i;
if(index > n/3 && index < 2*n/3)
a[index+ghosts] = 1.5;
else a[index+ghosts] = .5;
}
__syncthreads();

if(i==0){	//copy over for boundary conditions
for(int j = 0; j < ghosts; j++){
a[j] = a[j+n];
a[n+ghosts+j] = a[ghosts+j];
// a[j] = a[ghosts];
// a[n+ghosts+j] = a[n+ghosts-1];
}
// for(int z = 0; z < n+2*ghosts; z++){
// 	printf("%5d %10f\n", z, a[z]);
// }
}
}