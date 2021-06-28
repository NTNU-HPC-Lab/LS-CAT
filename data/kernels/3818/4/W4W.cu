#include "includes.h"
__global__ void W4W(int *w, int *out)
{
int tid = blockIdx.x;
int weight[sizeof(w)/sizeof(int)];
char c1 = (w[tid]/100)+48;
char c2 = ((w[tid]%100)/10)+48;
char c3 = w[tid]%10+48;
weight[tid]=(w[tid]/100)+ ((w[tid] % 100) / 10) + w[tid] % 10;
printf("%d, %c, %c, %c, %d\n", w[tid], c1, c2, c3, weight[tid]);
if (tid != 0)
{
if (weight[tid - 1] > weight[tid])
{
int x,y;
x=w[tid -1];y=w[tid];//collapsed to reserve pixels
out[tid -1]=y;out[tid]=x;
x=0;y=0;
x=weight[tid -1];y=weight[tid];
weight[tid -1]=y;weight[tid]=x;
}
if (weight[tid - 1] = weight[tid])
{

}
}
}