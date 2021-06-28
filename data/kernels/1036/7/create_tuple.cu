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




__global__ void create_tuple(double* devx, int* pos_x, int* pos_y, double* value, int N) {
int index = threadIdx.x;
if (index < N) {
pos_x[index] = index;
pos_y[index] = index;
value[index] = 2 * cosf(2 * devx[index]);
}
else if(index == N){
pos_x[index] = N;

}
}