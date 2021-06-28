#include "includes.h"

using namespace std;

//Check for edges valid to be part of augmented path

//Update frontier

__global__ void k2(const int N, bool* visited, int* frontier, bool* new_frontier) {
int count = 0;
for(int i=0;i<N;i++) {
if(new_frontier[i]) {
new_frontier[i] = false;
frontier[++count] = i;
visited[i] = true;
}
}
frontier[0] = count;
}