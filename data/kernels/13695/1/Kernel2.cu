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



__global__ void Kernel2(float *A,int N,int k){

int i = blockDim.x * blockIdx.x + threadIdx.x;
int j = blockDim.y * blockIdx.y + threadIdx.y;

__shared__ float k_k1,k1_k;
/*	example:
*	if we have to go from D -> F throw k and then k+1 we have to do:
*	DkF check
*  D(k+1)F check
*	Dk(k+1)F check
*	D(k+1)kF check
*	the min of these is the min dist.
*/
if(threadIdx.x==0 && threadIdx.y==0){
k_k1=A[k*N+(k+1)];
k1_k=A[(k+1)*N+k];
}
float x,y,asked,xn,yn;

asked=A[i*N+j];

x=A[k*N+j];
y=A[i*N+k];

// DkF
if(asked>x+y){
asked=x+y;
}

xn=A[i*N+(k+1)];
yn=A[(k+1)*N+j];

__syncthreads();

//	D(k+1)
if(xn>y+k_k1){
xn=y+k_k1;
}
//	(k+1)F
if(yn>x+k1_k){
yn=x+k1_k;
}
//	D(k+1)F or D(k+1)kF or Dk(k+1)F
if(asked>xn+yn){
asked=xn+yn;
}
//	min dist
A[i*N+j]=asked;
}