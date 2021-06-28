#include "includes.h"



using namespace std;

#define D 3
#define N 200
#define K 512
#define Nt 20
#define Rt 0.1f
#define c 0.001f
#define ct 0.0001f




__global__ void NextQTur(float* Qt, float* Pt) {
int i = threadIdx.x;
Qt[i + 0] += Pt[i + 0] * ct;
Qt[i + 1] += Pt[i + 1] * ct;
Qt[i + 2] += Pt[i + 2] * ct;
}