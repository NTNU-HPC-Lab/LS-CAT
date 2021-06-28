#include "includes.h"
__global__ void initSinusoid(float* a, float* x, float totalX, int n, int ghosts, float shift, float amp){
int i = threadIdx.x + blockDim.x*blockIdx.x;
for(int j = 0; blockDim.x*j + i < n; j++){
int index = j*blockDim.x+i;
float temp = 0;
for(int z = 0; z < index; z++){
temp += x[z+ghosts];
}
a[index+ghosts] = sinpi((temp/totalX)*2)*amp + shift;
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