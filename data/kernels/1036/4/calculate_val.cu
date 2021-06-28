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




__device__ void wait() {
for (int i = 1; i <= 10000000; i++);
}
__device__ double sqr(double x) {
return x * x;
}
__global__ void calculate_val(double* devx, double* val, int size) {
for (int index = blockIdx.x * blockDim.x + threadIdx.x;
index < size;
index += blockDim.x * gridDim.x)
{

int pre = index - 1;
if (pre < 0) pre += size;
int next = index + 1;
if (next >= size) next -= size;
val[index] = sqr(sin(devx[pre] * devx[index])) * sqr(sin(devx[next] * devx[index]));

}

//	wait();
}