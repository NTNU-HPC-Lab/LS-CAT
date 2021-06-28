#include "includes.h"
__global__ void blur(int* B,int* G,int* R, int* RB,int* RG,int* RR, int* K, int rows, int cols, int krows, int kcols) {

int index = blockIdx.x * 1024 + threadIdx.x;

if (index > rows*cols)
return;

int pixel_row = index/cols ;
int pixel_col = index - pixel_row*cols;

int pr,pc,idx;

int k_sum = 0;
int kr,kc;

int k_center_row = (krows-1)/2;
int k_center_col = (kcols-1)/2;

for(int i=0;i<krows;i++) {
for(int j=0;j<kcols;j++) {

kr = (i - k_center_row);
kc = (j - k_center_col);

pr = pixel_row + kr ;
pc = pixel_col + kc ;

idx = pr*cols + pc;

if (pr >=0 && pr < rows && pc>=0 && pc < cols) {
k_sum += K[kr*kcols + kc];

RB[index] += B[idx]*K[kr*kcols + kc];
RG[index] += G[idx]*K[kr*kcols + kc];
RR[index] += R[idx]*K[kr*kcols + kc];

}
}
}

RB[index] /= k_sum;
RG[index] /= k_sum;
RR[index] /= k_sum;
}