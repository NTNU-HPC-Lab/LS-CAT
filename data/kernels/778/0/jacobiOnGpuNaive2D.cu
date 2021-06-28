#include "includes.h"
#define DEBUG 0

__global__ void jacobiOnGpuNaive2D(double *MatA, double *d_MatC, int dim_x, int dim_y,int iter_max){
unsigned long long int i = blockDim.x * blockIdx.x + threadIdx.x;
unsigned long long int j = blockDim.y * blockIdx.y + threadIdx.y * 4;
unsigned long long int iPrev = i-1;
unsigned long long int iNext = i+1;
unsigned long long int jPrev = j-1;
unsigned long long int jNext = j+1;
unsigned long long int index = i * dim_y + j;
unsigned long long int indexUnroll3 = index - 1;
unsigned long long int indexUnroll2 = index - 2;
unsigned long long int indexUnroll1 = index - 3;
for (int k = 0; k < iter_max; k++){
if( i > 0 && j > 0 && i < (dim_x-1) && j <(dim_y-1)){
d_MatC[index] = 0.25f * (MatA[iPrev * dim_y + j] + MatA[iNext* dim_y + j] + MatA[i * dim_y+ jPrev] + MatA[i* dim_y + jNext]);
d_MatC[indexUnroll1] = 0.25f * (MatA[indexUnroll1 + 1] + MatA[indexUnroll1 - 1] + MatA[indexUnroll1 + dim_x] + MatA[indexUnroll1 - dim_x]);
d_MatC[indexUnroll2] = 0.25f * (MatA[indexUnroll2 + 1] + MatA[indexUnroll2 - 1] + MatA[indexUnroll2 + dim_x] + MatA[indexUnroll2 - dim_x]);
d_MatC[indexUnroll3] = 0.25f * (MatA[indexUnroll3 + 1] + MatA[indexUnroll3 - 1] + MatA[indexUnroll3 + dim_x] + MatA[indexUnroll3 - dim_x]);
if(DEBUG){
printf("index %llu %llu %llu %llu\n", index, indexUnroll3, indexUnroll2, indexUnroll1);
//printf("index %llu \n", index);
//printf("indexUnroll2 %llu, contornos %lf %lf %lf %lf \n",indexUnroll2, MatA[indexUnroll2 + 1], MatA[indexUnroll2 - 1], MatA[indexUnroll2 + dim_y] , MatA[indexUnroll2 - dim_y] );
}
}
__syncthreads();
if(index<dim_x*dim_y) MatA[index] = d_MatC[index];
}
}