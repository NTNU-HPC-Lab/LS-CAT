#include "includes.h"

using namespace std;

struct compressed_sparse_column {
int* data;
int* row;
int* column;
int* index_column;
int* index_row_start;
int* index_row_end;
};

struct graph {
compressed_sparse_column* dataset;
bool* roots;
bool* leaves;
bool* singletons;
int vertices;
int edges;
};

__global__ void pre_post_order(int* depth, int* zeta, int* zeta_tilde, graph* dataset_graph) {
int* pre = new int[dataset_graph->vertices];
int* post = new int[dataset_graph->vertices];

memset(pre, 0, dataset_graph->vertices * sizeof(int));
memset(post, 0, dataset_graph->vertices * sizeof(int));

bool* incoming_edges = new bool[dataset_graph->edges];
memset(incoming_edges, false, dataset_graph->edges * sizeof(bool));

bool* q = new bool[dataset_graph->vertices];
memcpy(q, dataset_graph->roots, sizeof(int) * dataset_graph->vertices);

while(true) {
bool* p = new bool[dataset_graph->vertices];
memset(p, false, dataset_graph->vertices * sizeof(bool));
bool global_check = false;

for(int i = 0; i < dataset_graph->vertices; i++) {
if( q[i] ) {
int pre_node = 	pre[i];
int post_node = post[i];

for(int j = dataset_graph->dataset->index_column[i]; dataset_graph->dataset->column[j] == i; j++) {
int neighbor_vertex = dataset_graph->dataset->row[j];
// zeta[i] = undefined!
pre[neighbor_vertex] = pre_node + zeta_tilde[neighbor_vertex];
post[neighbor_vertex] = post_node + zeta_tilde[neighbor_vertex];

incoming_edges[j] = true;
bool flag = true;
for(int k = 0; k < dataset_graph->edges; k++) {
if( dataset_graph->dataset->row[k] == neighbor_vertex && !incoming_edges[k] ) {
flag = false;
break;
}
}
if( flag ) {
global_check = true;
p[neighbor_vertex] = true;
}
}
pre[i] = pre_node + depth[i];
post[i] = post_node + (zeta[i] - 1);
}
}
q = p;
if( !global_check ) {
break;
}
}

}