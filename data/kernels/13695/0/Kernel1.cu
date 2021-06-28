#include "includes.h"

#define bufSize 700000


struct timeval startwtime,endwtime;

float *h_a;			// Table at host
float *d_a;			// Table at device
int tsize=0;		// number of rows or columns
size_t size = 0 ;	// size of table( tsize* tsize * sizeof(float*))
float* test;

void print(float *);
void make_table();
void serial();
void check();
void copytables();



__global__ void Kernel1(float *A,int N,int k){

int i = blockDim.x * blockIdx.x + threadIdx.x;
int j = blockDim.y * blockIdx.y + threadIdx.y;
//printf("Hello from %d %d \n",threadIdx.x,threadIdx.y);
if ( A[i*N+j] > A[i*N+k] + A[k*N+j] ){
A[i*N+j] = A[i*N+k] + A[k*N+j];
}
}