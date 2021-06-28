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




__global__ void sum_val(double* val, double* r) {
int index = threadIdx.x;
for (int i = 1; i < blockDim.x; i <<= 1) {
if (index % (i << 1) == i) {
val[index - i] += val[index];
}
__syncthreads();
}
if (index == 0) {
r[0] = val[0];
}
}