#include "includes.h"
__global__ void createFinalHiddenFeaturesKernel(const float *weights, const float *movie_rating_probs, float* final_hidden_feature_probs, int num_movies, int num_hidden_features) {

// weights[NUM_MOVIES][5][NUM_FEATURES]
// movie_rating_probs[NUM_MOVIES][5]
// final_hidden_feature_probs[NUM_FEATURES]
unsigned int hidden_id = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int movie_id = 0;
unsigned int rating = 0;
float dot_prod; // Temporary, local dot product variable
while (hidden_id < num_hidden_features) {
dot_prod = 0.00; // Initialize the dot product to 0

for (movie_id = 0; movie_id < num_movies; movie_id++) {
for (rating = 0; rating < 5; rating++) {
// Indexing: weights[movie_id][rating][feature_id]
// movie_id - [1, 17771]
// rating - [0, 4]
// hidden_id - [0, 99]
// Do the dot product
dot_prod += weights[movie_id*5*num_hidden_features
+ rating*num_hidden_features
+ hidden_id]
* final_hidden_feature_probs[hidden_id];
}
}
// Store the dot_product result
final_hidden_feature_probs[hidden_id] = dot_prod;

// Re-use this thread on another data point:
hidden_id += blockDim.x * gridDim.x;
}
}