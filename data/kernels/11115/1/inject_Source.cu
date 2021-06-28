#include "includes.h"

#define C0  0
#define CZ1 1
#define CX1 2
#define CY1 3
#define CZ2 4
#define CX2 5
#define CY2 6
#define CZ3 7
#define CX3 8
#define CY3 9
#define CZ4 10
#define CX4 11
#define CY4 12

__global__ void inject_Source(int id, int ii, float *p, float *_tableS, float *_sourceV, int *_locsS, int _dir, int _jt, int _ntSrc){
int ix = blockIdx.x * blockDim.x + threadIdx.x;
p[_locsS[ix]]+=(float)_dir/_jt * (
_tableS[ii + 0]*_sourceV[_ntSrc*ix+id]+
_tableS[ii + 1]*_sourceV[_ntSrc*ix+id+1]+
_tableS[ii + 2]*_sourceV[_ntSrc*ix+id+2]+
_tableS[ii + 3]*_sourceV[_ntSrc*ix+id+3]+
_tableS[ii + 4]*_sourceV[_ntSrc*ix+id+4]+
_tableS[ii + 5]*_sourceV[_ntSrc*ix+id+5]+
_tableS[ii + 6]*_sourceV[_ntSrc*ix+id+6]+
_tableS[ii + 7]*_sourceV[_ntSrc*ix+id+7]
);
}