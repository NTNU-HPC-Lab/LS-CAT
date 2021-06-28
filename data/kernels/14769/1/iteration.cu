#include "includes.h"

const int BLOCK_SIZE_X = 26;
const int BLOCK_SIZE_Y = 26;
const float w1 = 4.0/9.0, w2 = 1.0/9.0, w3 = 1.0/36.0;
const float Amp2 = 0.1, Width = 10, omega = 1;





__global__ void iteration(float* f_d, int ArraySizeX, int ArraySizeY)
{
int i;
int tx = threadIdx.x;
int ty = threadIdx.y;
int bx = blockIdx.x*(BLOCK_SIZE_X-2);
int by = blockIdx.y*(BLOCK_SIZE_Y-2);
int x = tx + bx;
int y = ty + by;
register float n,ux,uy,uxx,uyy,uxy,usq,Fx,Fy,Fxx,Fyy,Fxy,Fsq;
__shared__ float f_sh[BLOCK_SIZE_X][BLOCK_SIZE_Y][9];
//  __shared__ float f_p[BLOCK_SIZE_X][BLOCK_SIZE_Y][9];

for(i=0;i<9;i++)
f_sh[tx][ty][i]=f_d[x*ArraySizeY*9+y*9+ i];

__syncthreads();

n=f_sh[tx][ty][0]+f_sh[tx][ty][1]+f_sh[tx][ty][2]+f_sh[tx][ty][3]+f_sh[tx][ty][4]+f_sh[tx][ty][5]+f_sh[tx][ty][6]+f_sh[tx][ty][7]+f_sh[tx][ty][8];
ux=f_sh[tx][ty][1]-f_sh[tx][ty][2]+f_sh[tx][ty][5]-f_sh[tx][ty][6]-f_sh[tx][ty][7]+f_sh[tx][ty][8];
uy=f_sh[tx][ty][3]-f_sh[tx][ty][4]+f_sh[tx][ty][5]+f_sh[tx][ty][6]-f_sh[tx][ty][7]-f_sh[tx][ty][8];
ux/=n;
uy/=n;
uxx=ux*ux;
uyy=uy*uy;
uxy=2*ux*uy;
usq=uxx+uyy;
// implement the forcing terms and perform collision step
Fx=0;//Amp*sin(y*2*M_PI/cols);
Fy=0;
Fxx=2*n*Fx*ux;
Fyy=2*n*Fy*uy;
Fxy=2*n*(Fx*uy+Fy*ux);
Fsq=Fxx+Fyy;
Fx*=n;
Fy*=n;

f_sh[tx][ty][0]+=omega*(w1*n*(1-1.5*usq)-f_sh[tx][ty][0])-w1*1.5*Fsq;
f_sh[tx][ty][1]+=omega*(w2*n*(1+3*ux+4.5*uxx -1.5*usq)-f_sh[tx][ty][1])+w2*(3*Fx+4.5*Fxx-1.5*Fsq);
f_sh[tx][ty][2]+=omega*(w2*n*(1-3*ux+4.5*uxx -1.5*usq)-f_sh[tx][ty][2])+w2*(-3*Fx+4.5*Fxx-1.5*Fsq);
f_sh[tx][ty][3]+=omega*(w2*n*(1+3*uy+4.5*uyy -1.5*usq)-f_sh[tx][ty][3])+w2*(3*Fy+4.5*Fyy-1.5*Fsq);
f_sh[tx][ty][4]+=omega*(w2*n*(1-3*uy+4.5*uyy -1.5*usq)-f_sh[tx][ty][4])+w2*(-3*Fy+4.5*Fyy-1.5*Fsq);
f_sh[tx][ty][5]+=omega*(w3*n*(1+3*(ux+uy)+4.5*(uxx+uxy+uyy)-1.5*usq)-f_sh[tx][ty][5])+w3*(3*(Fx+Fy)+4.5*(Fxx+Fxy+Fyy)-1.5*Fsq);
f_sh[tx][ty][6]+=omega*(w3*n*(1+3*(-ux+uy)+4.5*(uxx-uxy+uyy)-1.5*usq)-f_sh[tx][ty][6])+w3*(3*(-Fx+Fy)+4.5*(Fxx-Fxy+Fyy)-1.5*Fsq);
f_sh[tx][ty][7]+=omega*(w3*n*(1+3*(-ux-uy)+4.5*(uxx+uxy+uyy)-1.5*usq)-f_sh[tx][ty][7])+w3*(3*(-Fx-Fy)+4.5*(Fxx+Fxy+Fyy)-1.5*Fsq);
f_sh[tx][ty][8]+=omega*(w3*n*(1+3*(ux-uy)+4.5*(uxx-uxy+uyy)-1.5*usq)-f_sh[tx][ty][8])+w3*(3*(Fx-Fy)+4.5*(Fxx-Fxy+Fyy)-1.5*Fsq);
__syncthreads();

//perfom stream step
if(tx>0 && tx<BLOCK_SIZE_X-1 && ty>0 && ty<BLOCK_SIZE_Y-1) {
f_d[x*ArraySizeY*9+y*9] = f_sh[tx][ty][0];//+omega*(w1*n*(1-1.5*usq)-f_sh[tx][ty][0])-w1*1.5*Fsq;
f_d[x*ArraySizeY*9+y*9+2] = f_sh[tx+1][ty][2];//+omega*(w2*n*(1-3*ux+4.5*uxx -1.5*usq)-f_sh[tx+1][ty][2])+w2*(-3*Fx+4.5*Fxx-1.5*Fsq);
f_d[x*ArraySizeY*9+y*9+1] = f_sh[tx-1][ty][1];//+omega*(w2*n*(1+3*ux+4.5*uxx -1.5*usq)-f_sh[tx-1][ty][1])+w2*(3*Fx+4.5*Fxx-1.5*Fsq);
f_d[x*ArraySizeY*9+y*9+4] = f_sh[tx][ty+1][4];//+omega*(w2*n*(1-3*uy+4.5*uyy -1.5*usq)-f_sh[tx][ty+1][4])+w2*(-3*Fy+4.5*Fyy-1.5*Fsq);
f_d[x*ArraySizeY*9+y*9+3] = f_sh[tx][ty-1][3];//+omega*(w2*n*(1+3*uy+4.5*uyy -1.5*usq)-f_sh[tx][ty-1][3])+w2*(3*Fy+4.5*Fyy-1.5*Fsq);
f_d[x*ArraySizeY*9+y*9+7] = f_sh[tx+1][ty+1][7];//+omega*(w3*n*(1+3*(-ux-uy)+4.5*(uxx+uxy+uyy)-1.5*usq)-f_sh[tx+1][ty+1][7])+w3*(3*(-Fx-Fy)+4.5*(Fxx+Fxy+Fyy)-1.5*Fsq);
f_d[x*ArraySizeY*9+y*9+5] = f_sh[tx-1][ty-1][5];//+omega*(w3*n*(1+3*(ux+uy)+4.5*(uxx+uxy+uyy)-1.5*usq)-f_sh[tx-1][ty-1][5])+w3*(3*(Fx+Fy)+4.5*(Fxx+Fxy+Fyy)-1.5*Fsq);
f_d[x*ArraySizeY*9+y*9+6] = f_sh[tx+1][ty-1][6];//+omega*(w3*n*(1+3*(-ux+uy)+4.5*(uxx-uxy+uyy)-1.5*usq)-f_sh[tx+1][ty-1][6])+w3*(3*(-Fx+Fy)+4.5*(Fxx-Fxy+Fyy)-1.5*Fsq);
f_d[x*ArraySizeY*9+y*9+8] = f_sh[tx-1][ty+1][8];//+omega*(w3*n*(1+3*(ux-uy)+4.5*(uxx-uxy+uyy)-1.5*usq)-f_sh[tx-1][ty+1][8])+w3*(3*(Fx-Fy)+4.5*(Fxx-Fxy+Fyy)-1.5*Fsq);

}

__syncthreads();
// apply periodi boundary conditions;
if(x == 0)
for(i = 0;i<9;i++)
f_d[x*ArraySizeY*9+y*9+i] = f_d[(ArraySizeX-2)*ArraySizeY*9+y*9+i];
if(x==ArraySizeX-1)
for(i=0;i<9;i++)
f_d[x*ArraySizeY*9+y*9+i] = f_d[ArraySizeY*9+y*9+i];
if(y == 0)
for(i = 0;i<9;i++)
f_d[x*ArraySizeY*9+y*9+i] = f_d[x*ArraySizeY*9+(ArraySizeY-2)*9+i];
if(y == ArraySizeY-1)
for(i =0;i<9;i++)
f_d[x*ArraySizeY*9 +y*9 +i] = f_d[x*ArraySizeY*9+9+i];

//   __syncthreads();
}