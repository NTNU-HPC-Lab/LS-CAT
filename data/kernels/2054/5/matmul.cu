#include "includes.h"
__device__ int inner_product(int p, int q, int *a, int c_a, int *b, int r_b, int c_b){
int i = p;
int j = q;
int tmp_ra = i * c_a;
int result = 0;

for(int x = 0; x < r_b; x++){
result += a[tmp_ra] * b[j];
tmp_ra += 1;
j += c_b;
}

return result;
}
__global__ void matmul(int *a, int c_a, int *b, int r_b, int c_b, int *c, int c_c, int N_BLOCKS, int N_THREADS){
int b_indx = blockIdx.x;
int t_indx = threadIdx.x;
int gindex = b_indx * N_THREADS + t_indx;
int i = gindex / c_c;
int j = gindex - i * c_c;
//int gindex = i * c_c  + j;
c[gindex] = inner_product(i, j, a, c_a, b, r_b, c_b);
__syncthreads();
}