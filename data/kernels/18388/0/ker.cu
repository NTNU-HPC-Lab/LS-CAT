#include "includes.h"
using namespace std;

long long remaining_N2(int , int ,long long );
long long remaining_N(int , int ,int );
__global__ void ker(float * cormat, float * upper,int n1,int n)
{
long idx = blockDim.x*blockIdx.x+threadIdx.x;
long i = idx%n1;
long j = idx/n1;
if(i<j && i<n1 && j<n)
{
long tmp=i;
tmp*=(i+1);
tmp/=2;
long tmp_2=i;
tmp_2*=n;
tmp_2=tmp_2-tmp;
tmp_2+=j;
tmp_2-=i;


upper[tmp_2-1]=cormat[j*n+i];
}
}