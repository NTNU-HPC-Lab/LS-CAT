#include "includes.h"
//double* x, * devx, * val, * gra, * r, * graMax;
//double* hes_value;
////int size;
//int* pos_x, * pos_y;
//int* csr;
double* x;
//thrust::pair<int, int> *device_pos;
//typedef double (*fp)(double);
//typedef void (*val_fp)(double*, double*, int);
//typedef void (*valsum_fp)(double*, double*,int);
//typedef void (*gra_fp)(double*, double*, int);
//typedef void (*gramin_fp)(double*, double*,int);
//typedef void (*hes_fp)( double*, thrust::pair<int, int>*, double*, int);
//typedef void (*print_fp)(double*, int);
int numSMs;




__device__ __host__ inline double Max(double x, double y) {
x = fabs(x);
y = fabs(y);
return x > y ? x : y;
}
__global__ void max_gra(double* gra, double* max) {
int index = threadIdx.x;
for (int i = 1; i < blockDim.x; i <<= 1) {
if (index % (i << 1) == i) {
gra[index - i] = Max(gra[index - i], gra[index]);
}
__syncthreads();
}
if (index == 0) {
max[0] = gra[0];
}

}