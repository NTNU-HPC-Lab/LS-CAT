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


__global__ void core_svd2D_recomp(float *Y, float *E, float * V, size_t N){   // N is NOT the total size, but only the size excluding the last dimension (of size 3)
size_t idd=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);
if(idd>=N) return;
int k;
double ee[2];
double vv[2];
double tmp[3];

for (k=0;k<2;k++){   // get the eigenvalues and eigenvectors
ee[k]=E[idd+N*k];
vv[k]=V[idd+N*k];
}
tmp[0]=ee[0]*pow(vv[0],2) + ee[1]*pow(vv[1],2);
tmp[1]=vv[0]*vv[1]*(ee[0]-ee[1]);
tmp[2]=ee[0]*pow(vv[1],2)+ee[1]*pow(vv[0],2);
for (k=0;k<3;k++){  // set result
Y[idd+N*k]=tmp[k];
}
}