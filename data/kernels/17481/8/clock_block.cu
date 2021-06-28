#include "includes.h"
__global__ void clock_block(clock_t *d, clock_t clock_count) {
clock_t start_clock = clock64();
clock_t clock_offset = 0;
while (clock_offset < clock_count) {
clock_offset = clock64() - start_clock;
}
if (d) {
*d = clock_offset;
}
}