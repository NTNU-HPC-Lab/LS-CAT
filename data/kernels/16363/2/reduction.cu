#include "includes.h"
__global__ void reduction(bool *B,int *number,int order){
int num = 0;
int idx = blockIdx.x * blockDim.x + threadIdx.x;

//if(idx==0) printf("ORDER%d\n",order);
//printf("IDX%d\n",idx);
if(idx<order){

for(int i = 0 ; i<order; i++)
if(B[idx*order+i]==1)
num ++; //= B[idx*order + i];
//if(B[idx]==1)
//printf("CUDANUM%d\n",num);
number[idx] = num;
//atomicAdd(number,num);
//printf("NUMBER%d\n",number);//<<endl;
}
}