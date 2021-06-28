#include "includes.h"



using namespace std;

#define D 3
#define N 200
#define K 512
#define Nt 20
#define Rt 0.1f
#define c 0.001f
#define ct 0.0001f




__global__ void Sqrt(float* Q, float* P, float* Qt, float* Pt, float* Eg, float* Epg) {
int x = blockIdx.x;
int y = threadIdx.x;
int i = x * K * D + y * D;
//int z = threadIdx.z;
//printf("I = %i \n", x);
for (int j = 0; j < 3; j++) {
Q[i + j] = 0.01;
Qt[i + j] = 0.6;
P[i + j] = 0.3;
Pt[i + j] = 0.5;
Epg[i / D ] = 100000;
Eg[i / D ] = 0.5;
}
}