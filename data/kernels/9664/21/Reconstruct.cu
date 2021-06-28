#include "includes.h"
__global__ void Reconstruct(int *nex, unsigned long nextsize, double4 *pc, double4 *vc, double4 *a3, double4 *a, double4 *a1, double4 *a2, double4 *pva3, double4 *aaa) {


unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;

int k = gtid/nextsize;
int who = nex[gtid - k*nextsize];

if(gtid<nextsize){
pc[who] = pva3[gtid];
}
else if(gtid >= nextsize && gtid < 2*nextsize){
vc[who] = pva3[gtid];
}
else if(gtid >= 2*nextsize && gtid < 3*nextsize){
a3[who] = pva3[gtid];
}
else if(gtid >= 3*nextsize && gtid < 4*nextsize){
a[who] = aaa[gtid - 3*nextsize];
}
else if(gtid>= 4*nextsize && gtid < 5*nextsize){
a1[who] = aaa[gtid - 3*nextsize];
}
else if(gtid>= 5*nextsize && gtid < 6*nextsize){
a2[who] = aaa[gtid - 3*nextsize];
}


}