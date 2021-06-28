#include "includes.h"
// cuda ÇÁ·Î±×·¡¹Ö Ã¹ ½ÃÀÛÀ» ÇÏ±â À§ÇÑ ±âº» default ÄÚµå¸¦ °¡Á®¿Ô´Ù.
// ¼³¸íÀº ¾ÆÁ÷ÀÌ´Ï ¿ì¼± c++½ºÅ¸ÀÏ ÄÚµùÀÌ³ª ÀÍÈ÷°í, SIZE ¼ýÀÚ¸¦ ¹Ù²ã°¡¸ç ½ÇÇàÇÑ °á°ú¸¦ »ìÆìº¸ÀÚ
// µüÈ÷ ¿©±â¼­ ¹è¿ï °ÍÀº ¾ø°í cuda_main2.cu ºÎÅÍ ÇÏ³ª¾¿ Â÷±ÙÂ÷±Ù Â¤¾îº¼ ¿¹Á¤

#define SIZE 1024

// __global__À» ÅëÇØ¼­ Ä¿³ÎÀÓÀ» Ç¥½ÃÇÑ´Ù. host¿¡¼­ È£ÃâµÈ´Ù.

__global__ void VectorAdd(int *a, int *b, int *c, int n) {
// ¼ö¸¹Àº ½º·¹µå°¡ µ¿½Ã¿¡ Ã³¸®ÇÑ´Ù.
// µû¶ó¼­ threadIdx(½º·¹µå ÀÎµ¦½º)¸¦ ÅëÇØ¼­ ½º·¹µåµéÀ» ±¸º°ÇÑ´Ù.
int i = threadIdx.x;

printf("threadIdx.x : %d, n : %d\n", i, n);

for (i = 0; i < n; i++) {
c[i] = a[i] + b[i];
printf("%d = %d + %d\n", c[i], a[i], b[i]);
}
}