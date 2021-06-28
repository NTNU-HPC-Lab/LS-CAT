#include "includes.h"
__global__ void kernel_256_winograd_AtIA(float *pInputs, float *pBiases, float *pScales, float *pOutputs) {
int Tilex = blockIdx.x, Tiley = blockIdx.y, Iny = threadIdx.y, kz = blockIdx.z, Inx = threadIdx.x;
int c_input = Inx*6 + Iny;

__shared__ float bias, scale;
extern __shared__ float input[];

input[c_input] = pInputs[c_input*16*256 + (Tilex*4+Tiley)*256 + kz];
bias = pBiases[kz];
scale = pScales[kz];
__syncthreads();

float tmp = 0;
switch(Inx) {
case 0:
tmp = input[Iny] + input[6+Iny] + input[12+Iny] + input[18+Iny] + input[24+Iny];
break;
case 1:
tmp = input[6+Iny] - input[12+Iny] + 2*input[18+Iny] - 2*input[24+Iny];
break;
case 2:
tmp = input[6+Iny] + input[12+Iny] + 4*input[18+Iny] + 4*input[24+Iny];
break;
case 3:
tmp = input[6+Iny] - input[12+Iny] + 8*input[18+Iny] - 8*input[24+Iny] + input[30+Iny];
break;
}
__syncthreads();

input[c_input] = tmp;
__syncthreads();

if (Inx > 3 || (Tilex == 3 && Inx > 1)) return;

int x;
float o;
switch(Iny) {
case 0:
x = Inx*6;
o = scale*(input[x]+input[x+1]+input[x+2]+input[x+3]+input[x+4]) + bias;
pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+1)*256 + kz] = o > 0 ? o : 0;
break;
case 1:
x = Inx*6;
o = scale*(input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]) + bias;
pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+2)*256 + kz] = o > 0 ? o : 0;
break;
case 2:
if (Tiley == 3) break;
x = Inx*6;
o = scale*(input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]) + bias;
pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+3)*256 + kz] = o > 0 ? o : 0;
break;
case 3:
if (Tiley == 3) break;
x = Inx*6;
o = scale*(input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]) + bias;
pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+4)*256 + kz] = o > 0 ? o : 0;
break;
}
}