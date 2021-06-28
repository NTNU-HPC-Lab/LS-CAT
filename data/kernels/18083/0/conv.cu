#include "includes.h"
__global__ void conv(fmap *input,int *ip,int *weights,int R,int S,fmap *output, int Sx, int Sy,int *op,int Px,int Py){
unsigned int input_id = (blockIdx.x*gridDim.y + blockIdx.y + blockIdx.z*gridDim.x*gridDim.y)*blockDim.x + threadIdx.x;
int C,H,W,M,E,F;
//N = input->dim1;
C = input->dim2;
H = input->dim3;
W = input->dim4;
M = output->dim2;
E = output->dim3;
F = output->dim4;
H+=2*Py;
W+=2*Px;
/*unsigned int weight_id = input_id%(C*R*S);
int a = weight_id/(R*S);
weight_id = weight_id%(R*S);
int b = weight_id/S;
int c = weight_id%S;*/
int i = input_id/(M*E*F*C*R*S);
input_id = input_id%(M*E*F*C*R*S);
int j = input_id/(E*F*C*R*S);
input_id = input_id%(E*F*C*R*S);
int k = input_id/(F*C*R*S);
input_id = input_id%(F*C*R*S);
int l = input_id/(C*R*S);
input_id = input_id%(C*R*S);
int m = input_id/(R*S);
input_id = input_id%(R*S);
int n = input_id/S;
int o = input_id%S;

int temp = (*(ip + i*C*H*W + m*H*W + (k*Sy + n)*W + (l*Sx + o)))*(*(weights + j*C*R*S + m*R*S + n*S + o));
atomicAdd((op + i*M*E*F + j*E*F + k*F + l), temp);

/* printf("Input fmap\n");
printf("%d %d %d %d\n",N,C,H,W);
for(int i=0;i<N;i++){
for(int j=0;j<C;j++){
for(int k=0;k<H;k++){
for(int l=0;l<W;l++)
printf("%3d ",ip[i*C*H*W + j*H*W + k*H + l]);
printf("\n");
}
printf("\n\n");
}
printf("\n\n\n");
}

printf("Weight fmap\n");
printf("%d %d %d %d\n",M,C,R,S);
for(int i=0;i<M;i++){
for(int j=0;j<C;j++){
for(int k=0;k<R;k++){
for(int l=0;l<S;l++)
printf("%3d ",weights[i*C*R*S + j*R*S + k*S + l]);
printf("\n");
}
printf("\n\n");
}
printf("\n\n\n");
}*/

}