#include "includes.h"
__global__ void yuan(const char *text, int *pos, int text_size) {
int textP = blockIdx.x * blockDim.x + threadIdx.x;
if (textP >= text_size) return;
const char *start = text + textP;
while (start >= text && *start > ' ') {
start--;
}
pos[textP] = text + textP - start;

}