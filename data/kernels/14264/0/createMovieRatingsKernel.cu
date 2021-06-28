#include "includes.h"
__global__ void createMovieRatingsKernel(const float *weights, const float *initial_hidden_feature_probs, float* movie_rating_probs, int num_movies, int num_hidden_features) {

// weights[NUM_MOVIES][5][NUM_FEATURES]
// initial_hidden_feature_probs[NUM_FEATURES]
// final_movie_ratings[NUM_MOVIES][5]
//
// movie_rating_index = movie_id * 5 + rating_id
//      (index of current movie_id/rating_id pair)
unsigned int movie_rating_id = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int hidden_id = 0;
float dot_prod; // Temporary, local dot product variable
while (movie_rating_id < num_movies * 5) {
dot_prod = 0.00; // Initialize the dot product to 0

for (hidden_id = 0; hidden_id < num_hidden_features; hidden_id++) {
// Indexing: weights[movie_id][rating][feature_id]
// movie_id - [1, 17771]
// rating - [0, 4]
// feature_id - [0, 99]
// Do the dot product
dot_prod += weights[movie_rating_id*num_hidden_features
+ hidden_id]
* initial_hidden_feature_probs[hidden_id];
}
// Store the dot_product result
movie_rating_probs[movie_rating_id] = dot_prod;

// Re-use this thread on another data point:
movie_rating_id += blockDim.x * gridDim.x;
}
}