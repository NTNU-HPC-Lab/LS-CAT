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




__device__ double sqr(double x) {
return x * x;
}
__global__ void calculate_gra(double* devx, double* gra,int size) {
for (int index = blockIdx.x * blockDim.x + threadIdx.x;
index < size;
index += blockDim.x * gridDim.x)
{
int pre = index - 1;
if (pre < 0) pre += size;
int next = index + 1;
if (next >= size) next -= size;
gra[index] = devx[pre] * sin(2.0 * devx[index] * devx[pre]) + devx[next] * sin(2.0 * devx[index] * devx[next]);
printf("gra %d %d %d %f %f %f\n", pre, index, next, sqr(devx[index]), devx[pre] * sin(2.0 * devx[index] * devx[pre]), gra[index]);
}
}