#include "includes.h"



__global__ void kernelGPU(float *R,float* G,float* B,float* Rin,float*Gin,float*Bin,int M,int N,int L){

int tId= threadIdx.x+blockIdx.x*blockDim.x;
int i;
if(tId<M*N){
R[tId]=0;
G[tId]=0;
B[tId]=0;
for(i=0; i<L; ++i ){

R[tId]+= Rin[tId+i*M*N];
G[tId]+= Gin[tId+i*M*N];
B[tId]+= Bin[tId+i*M*N];
}

R[tId]=R[tId]/L;
G[tId]=G[tId]/L;
B[tId]=B[tId]/L;
}


}