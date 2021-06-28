#include "includes.h"
__device__ float colellaEvenInterp(float*in){
return (7.0/12)*(*(in+1) - *(in-1)) - (1.0/12)*((*(in+2) + *(in-1))-(*(in+1) + *(in-2)));
}
__device__ float linInterp(float* in){	//dangerous function, need to make sure you're only using it on the in-bounds parts of array
return ((*(in+1) + *in)/2) - ((*in + *(in-1))/2);
}
__global__ void advect(float* a, float* oA, float* x, float u, int n, int ghosts, float tmax){
__shared__ float dt;
__shared__ float minDx;
__shared__ float timeElapsed;
__shared__ int counter;
// __shared__ bool* areYouLessThan;

int i = threadIdx.x + blockDim.x*blockIdx.x;
timeElapsed = 0;

if(i == 0){
minDx = x[0];
dt = (minDx/u)/500;
counter = 0;
}
__syncthreads();

while(timeElapsed < tmax){
// if(i==0){
// 	minDx = x[0+ghosts];
// 	areYouLessThan = new bool[n];
// }
// __syncthreads();

// for(int j = 0; blockDim.x*j + i < n; j++){
// 	int index = j*blockDim.x+i;
// 	if(x[index+ghosts] < minDx)
// 		areYouLessThan[index] = true;
// 	else
// 		areYouLessThan[index] = false;
// }

// __syncthreads();

// if(i == 0){
// 	for(int j = 0; j < n; j++){
// 		if(areYouLessThan[j]){
// 			if(x[j+ghosts] < minDx)
// 				minDx = x[j+ghosts];
// 		}
// 	}

// dt = (minDx/u)/1000;
// 	// printf("dt: %f\n", dt);
// 	delete[] areYouLessThan;
// }

// __syncthreads();



for(int j = 0; blockDim.x*j + i < n; j++){
int index = j*blockDim.x+i+ghosts;
oA[index] = a[index] - dt*u*colellaEvenInterp(&a[index])/x[index];
// oA[index] = a[index] - dt*u*linInterp(&a[index])/x[index];
// printf("%d %f %f\n", index, a[index], oA[index]);
a[index] = oA[index];
}

__syncthreads();
// printf("%d here\n", i);
if(counter == 0){
for(int j = 0; blockDim.x*j + i < n; j++){
int index = j*blockDim.x+i+ghosts;
printf("%10f\t%10d\t%f\n", timeElapsed, index-ghosts, a[index]);
}
}
if(i==0){	//copy over for boundary conditions
for(int j = 0; j < ghosts; j++){
a[j] = a[j+n];
a[n+ghosts+j] = a[ghosts+j];
// a[j] = a[ghosts];
// a[n+ghosts+j] = a[n+ghosts-1];
}

// printf("%f\n",dt);
// printf("%p %f\n", dtt, *dtt);
timeElapsed += dt;
counter++;
if(counter == 10000)
counter = 0;
}
}
}