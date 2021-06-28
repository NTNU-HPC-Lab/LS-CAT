#include "includes.h"
__global__ void p2pPingPongLatencyTest( void *_pLocal, void *_pRemote, uint64_t *pTimestamps, int bWait, int cIterations )
{
volatile int *pLocal = (volatile int *) _pLocal;
volatile int *pRemote = (volatile int *) _pRemote;
int pingpongValue = 0;
while ( cIterations-- ) {
*pTimestamps++ = clock64();
if ( bWait )
while ( *pLocal != pingpongValue );
bWait = 1;
pingpongValue = 1-pingpongValue;
*pRemote = pingpongValue;
}
}