#include "includes.h"
#define CUDAMAXTHREADPERBLOCK 1024
#define CUDAMAXBLOCK 65536

using namespace std;

__global__ void kernelTestOpenmp(int *dev_b, int tt){
for (int i = 0; i < tt; i++) {
if (dev_b[i] != i) {
printf("no!!!");
}
printf("yes!!!!");
}
}