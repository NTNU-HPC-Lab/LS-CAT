#include "includes.h"
/************************* CudaMat ******************************************
*   Copyright (C) 2008-2009 by Rainer Heintzmann                          *
*   heintzmann@gmail.com                                                  *
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; Version 2 of the License.               *
*                                                                         *
*   This program is distributed in the hope that it will be useful,       *
*   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
*   GNU General Public License for more details.                          *
*                                                                         *
*   You should have received a copy of the GNU General Public License     *
*   along with this program; if not, write to the                         *
*   Free Software Foundation, Inc.,                                       *
*   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
***************************************************************************
* Compile with:
* Windows:
system('"c:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin\vcvars32.bat"')
system('nvcc -c cudaArith.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin')

Window 64 bit:
system('nvcc -c cudaArith.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin" -I"c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include" ')

Linux:
* File sudo vi /usr/local/cuda/bin/nvcc.profile
* needs the flag -fPIC  in the include line
system('nvcc -c cudaArith.cu -v -I/usr/local/cuda/include/')
*/

// To suppress the unused variable argument for ARM targets
#pragma diag_suppress 177


#ifndef NAN   // should be part of math.h
#define NAN (0.0/0.0)
#endif

#define ACCU_ARRTYPE double  // Type of the tempory arrays for reduce operations
#define IMUL(a, b) __mul24(a, b)

//#define BLOCKSIZE 512
//#define BLOCKSIZE 512
// below is blocksize for temporary array for reduce operations. Has to be a power of 2 in size
#ifndef CUIMAGE_REDUCE_THREADS  // this can be defined at compile time via the flag NVCCFLAG='-D CUIMAGE_REDUCE_THREADS=512'
#define CUIMAGE_REDUCE_THREADS 512
#endif
// (prop.maxThreadsPerBlock)
// #define CUIMAGE_REDUCE_THREADS 512
// #define CUIMAGE_REDUCE_THREADS 128
//#define CUIMAGE_REDUCE_BLOCKS  64

#define NBLOCKS(N,blockSize) (N/blockSize+(N%blockSize==0?0:1))

#define NBLOCKSL(N,blockSize) 1
// min((N/blockSize+(N%blockSize==0?0:1)),prop.maxGridSize[0])


__global__ void core_svd2D(float *X, float *Ye, float * Yv, size_t N){   // N is NOT the total size, but only the size excluding the last dimension (of size 3)
size_t idd=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);
if(idd>=N) return;
int k;
double n;
double tmp[3];
double E[2];
double U[2];
double trace;
double delta;

for (k=0;k<3;k++)   // get the matrix value [X(1,1) X(2,1)=X(1,2), X(2,2)]
tmp[k]=X[idd+N*k];

if (fabs(tmp[1]) < 1e-15){
E[0]=tmp[0];
E[1]=tmp[2];
U[0]=1.0;
U[1]=0.0;
}
else{
trace=tmp[0]+tmp[2];
delta=(tmp[0]-tmp[2])*(tmp[0]-tmp[2])+4*tmp[1]*tmp[1];
E[0]=0.5*(trace+sqrt(delta));
E[1]=0.5*(trace-sqrt(delta));
n=sqrt((E[0]-tmp[0])*(E[0]-tmp[0])+tmp[1]*tmp[1]);
U[0]=tmp[1]/n;
U[1]=(E[0]-tmp[0])/n;
}

for (k=0;k<2;k++){  // set result
Ye[idd+N*k]=E[k];
Yv[idd+N*k]=U[k];
}
}