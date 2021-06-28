#include "includes.h"
using namespace std;

long long remaining_N2(int , int ,long long );
long long remaining_N(int , int ,int );
__global__ void ker2(float * cormat, float * upper,int n1,int n,long long upper_size,int N,int i_so_far,long long M1)
{
long long idx = blockDim.x;
idx*=blockIdx.x;
idx+=threadIdx.x;
long i = idx/n;
long j = idx%n;

if(i<j && i<n1 && j<n)// &&i<N &&j<N && idx<(n1*n))
{
long long tmp=i;
tmp*=(i+1);
tmp/=2;
long long tmp_2=i;
tmp_2*=n;
tmp_2=tmp_2-tmp;
tmp_2+=j;
tmp_2-=i;
long long indexi=n1;
indexi*=j;
indexi=indexi+i;
upper[tmp_2-1]=cormat[indexi];
//if((i==39001 &&j == 69999)||(i==1 && j==2))
// printf("\n\n\n thread:  %f ",upper[tmp_2-1]," ",cormat[indexi]);
}

}