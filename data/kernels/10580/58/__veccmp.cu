#include "includes.h"
__global__ void __veccmp(int *a, int *b, int *d) {
int xa = *a;
int xb = *b;
int xc = 0;
int xd = 0;
asm("vset4.s32.s32.ne" "%0, %1.b0000, %2, %3;": "=r" (xd) : "r" (xa), "r" (xb), "r" (xc));
*d++ = xd;
asm("vset4.s32.s32.ne" "%0, %1.b1111, %2, %3;": "=r" (xd) : "r" (xa), "r" (xb), "r" (xc));
*d++ = xd;
asm("vset4.s32.s32.ne" "%0, %1.b2222, %2, %3;": "=r" (xd) : "r" (xa), "r" (xb), "r" (xc));
*d++ = xd;
asm("vset4.s32.s32.ne" "%0, %1.b3333, %2, %3;": "=r" (xd) : "r" (xa), "r" (xb), "r" (xc));
*d = xd;
}