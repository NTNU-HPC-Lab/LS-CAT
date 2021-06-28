#include "includes.h"
/***
__shared__  通过共享内存来完成线程间的通信
这一段代码 通过共享内存
***/


cudaError_t addWithCuda(int *c, const int *a, size_t size);


__global__ void addKernel(int *c, const int *a){
int i = threadIdx.x;
extern __shared__ int seme []; //声明一个全局的 共享内存的变量
seme[i] = a[i];
__syncthreads();  //同一个块的线程同步  等待seme将所有数据加载进来
if(i==0){ //第一个线程进行二次方
c[0] = 0;
for (int d=0; d<5; d++){
printf("seme[d] * seme [d] %d \n", d);
c[0] += seme[d] * seme [d];
}
printf("给 seme 赋值 %d ", i);
seme[i] = 0;
}
if(i==1){
c[1] = 0;
for (int d=0; d<5; d++){
printf("c[1] += seme[d] %d \n", d);
c[1] += seme[d];
}
printf("给 seme 赋值 %d ", i);
seme[i] = 0;
}
if(i==2){
c[2] = 1;
for(int d=0; d<5; d++){
printf("c[2] *= seme[d] %d \n", d);
c[2] *= seme[d];
}
printf("给 seme 赋值 %d ", i);
seme[i] = 0;
}
}