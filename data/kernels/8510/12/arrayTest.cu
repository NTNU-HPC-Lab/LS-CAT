#include "includes.h"
__global__ void arrayTest(int n, long *factor, long *arr, long *result, int *const_arr1, long *const_arr2)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i == 0) {
/*
printf("In ArrayTest n=%d factor=%p arr=%p result=%p \n",n,factor,arr,result);
printf("In const %d %d %d\n",const_arr1[0],const_arr1[1],const_arr1[2]);
printf("In const %ld %ld %ld\n",const_arr2[0],const_arr2[1],const_arr2[2]);
*/
}

if (i<n)
{
int idx = i * 3;
result[idx]=arr[idx] * factor[i];
result[idx + 1]=arr[idx + 1] * factor[i];
result[idx + 2]=arr[idx + 2] * factor[i];
/*
printf("ArrayTest  [%ld] * [%ld %ld %ld] = [%ld %ld %ld] \n", factor[i],
arr[idx],arr[idx+1],arr[idx+2],
result[idx],result[idx+1],result[idx+2]);
*/
}

}