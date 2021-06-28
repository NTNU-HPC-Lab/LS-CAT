#include "includes.h"
__global__ void lineark(int *ip,int *weight,int *op,int N,int M,int L){
unsigned int input_id = (blockIdx.x*gridDim.y + blockIdx.y + blockIdx.z*gridDim.x*gridDim.y)*blockDim.x + threadIdx.x;
int i = input_id/(M*L);
input_id = input_id%(M*L);
int j = input_id/L;
int k = input_id%L;

int temp = (*(ip + i*L + k))*(*(weight + j*L + k));
atomicAdd((op + i*M + j),temp);
}