#include "includes.h"
#define N 15000

using namespace std;
//Çäåñü: threadIdx.x  èäåíòèôèêàòîð ïîòîêà â áëîêå ïî êîîðäèíàòå x,
//blockIdx.x  èäåíòèôèêàòîð áëîêà â ãðèäå ïî êîîðäèíàòå x,
//blockDim.x  êîëè÷åñòâî ïîòîêîâ â îäíîì áëîêå.

__global__ void MatrVectMul(int *d_c, int *d_a, int *d_b)
{
int i = blockIdx.x*blockDim.x+threadIdx.x;
if(i<N)
{
d_c[i]=0;
for (int k=0;k<N;k++)
d_c[i]+=d_a[i+k*N]*d_b[k];
}
}