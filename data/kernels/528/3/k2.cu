#include "includes.h"
__global__ void k2(const int N, bool* visited, int* frontier, bool* new_frontier, bool* augFound) {
int count = 0;
for(int i=0;i<N;i++) {
if(new_frontier[i]) {
new_frontier[i] = false;
frontier[++count] = i;
visited[i] = true;
}
}
frontier[0] = count;

//Complete search if sink has been reached
for(int i = 0; i < frontier[0]; i++)
if(frontier[i + 1] == (N - 1))
augFound[0] = true;
}