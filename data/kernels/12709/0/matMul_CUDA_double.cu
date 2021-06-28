#include "includes.h"

#define NUM_THREAD  256  // Number of thread blocks
#define print(x) printf("%d",x)

float *matrixMul_float_serial(float vector1[], float vector2[], int size);
float *matrixMul_float_parallel(float vector1[], float vector2[], int size, int thread_count);
float *matrixMul_float_cuda(float* vector1, float* vector2, int num);
double *matrixMul_double_serial(double vector1[], double vector2[], int size);
double *matrixMul_double_parallel(double vector1[], double vector2[], int size, int thread_count);
double *matrixMul_double_cuda(double* vector1, double* vector2, int num);
double doubleGen();
float floatGen();
void operations(int size, int parallel, int serial, int cuda, int verify, int thread_count);
void print_results_float( int size, double time_spent);
void print_results_double( int size, double time_spent);
double verifyVectord(double *vector1, double *vector2, int size);
float verifyVectorf(float *vector1, float *vector2, int size);



__global__ void matMul_CUDA_double(double *sum, int size, double *vector1, double *vector2){
int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
int k;
if(idx < size*size){
for(k=0; k< size; k++){
sum[idx] += (*(vector1+(idx-(idx % size)+k))) * (*(vector2+(k*size+(idx % size))));
}
}
}