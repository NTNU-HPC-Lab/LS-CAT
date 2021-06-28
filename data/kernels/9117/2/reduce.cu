#include "includes.h"

float *A,*L,*U,*input;
void arrayInit(int n);
void verifyLU(int n);
void updateLU(int n);
void freemem(int n);

/*
*/


__global__ void reduce( float *a, int size, int c) {
int tid = blockIdx.x;	//Handle the data at the index
int thid = threadIdx.x;
int index=c,j=0;//size=b
int numthreads = blockDim.x;
for(j=index+1;j<size;j+=numthreads) {
a[((tid+index+1)*size + j+thid)] = (float)(a[((tid+index+1)*size + j+thid)] - (float)a[((tid+index+1)*size+index)] * a[((index*size) + j+thid)]);
}

}