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




__global__ void minus_gra(double* gra,int size) {
for (int index = blockIdx.x * blockDim.x + threadIdx.x;
index < size;
index += blockDim.x * gridDim.x)
{
gra[index]=0.0-gra[index];
}
}