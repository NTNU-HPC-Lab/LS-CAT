#include "includes.h"
__global__ void computeMoment(int *readArr, int *writeArr, double *weightArr, int n){
// The dimensions are hardcoded here to simplify extra syntax
// cuda uses for dynamic shared memory allocation
__shared__ int readArr_shared[32][32];
__shared__ double weightArr_shared[5][5];

int row = blockIdx.x*blockDim.x + threadIdx.x;
int col = blockIdx.y*blockDim.y + threadIdx.y;

if(threadIdx.x<5 && threadIdx.y < 5){
weightArr_shared[threadIdx.x][threadIdx.y] = weightArr[threadIdx.x*WINDOW_SIZE + threadIdx.y];
}
__syncthreads();

// Only values within the below borders will be used but the __syncthreads()
// function has to be called outside if statements so we load everything here
readArr_shared[threadIdx.x][threadIdx.y] = readArr[row*n + col];
__syncthreads();

// If coordinates are between boundaries
// update the write array accordingly
if(row < n && col < n){
float influence = 0.0f;
for (int i=-2; i<3; i++)
{
for (int j=-2; j<3; j++)
{
//add extra n so that modulo behaves like mathematics modulo
//that is return only positive values
if(threadIdx.x >= MIN_MARGIN && threadIdx.y >= MIN_MARGIN &&
threadIdx.x <= 31-MIN_MARGIN && threadIdx.y <= 31-MIN_MARGIN){
int y = threadIdx.x + i;
int x = threadIdx.y + j;
influence += weightArr_shared[i+2][j+2]*readArr_shared[y][x];
}else{
int y = (row+i+n)%n;
int x = (col+j+n)%n;
influence += weightArr_shared[i+2][j+2]*readArr[y*n + x];
}
}
}

if(threadIdx.x >= MIN_MARGIN && threadIdx.y >= MIN_MARGIN &&
threadIdx.x <= 31-MIN_MARGIN && threadIdx.y <= 31-MIN_MARGIN){
writeArr[row*n + col] = readArr_shared[threadIdx.x][threadIdx.y];
if 		(influence<-diff)	writeArr[row*n + col] = -1;
else if (influence>diff)	writeArr[row*n + col] = 1;
}else {
writeArr[row*n + col] = readArr[row*n + col];
if 		(influence<-diff)	writeArr[row*n + col] = -1;
else if (influence>diff)	writeArr[row*n + col] = 1;
}
}
__syncthreads();

}