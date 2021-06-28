#ifndef ANIMESH_KERS_HPP_
#define ANIMESH_KERS_HPP_

#include "cuda_utils.hpp"
#include "transfo.hpp"
#include "mesh.hpp"
#include "skeleton.hpp"
#include "animesh_enum.hpp"

/** @namespace Kernels
    @brief The cuda kernels used to animate the mesh

    @see Animesh
 */

// =============================================================================
namespace Animesh_kers{
// =============================================================================

using namespace Cuda_utils;

/// Computes the potential at each vertex of the mesh. When the mesh is
/// animated, if implicit skinning is enabled, vertices move so as to match
/// that value of the potential.
__global__ void
compute_base_potential(Skeleton_env::Skel_id skel_id,
                       const Point_cu* d_input_vertices,
                       const int nb_verts,
                       float* d_base_potential);

/// Match the base potential after basic ssd deformation
/// (i.e : do the implicit skinning step)
__global__
void match_base_potential(Skeleton_env::Skel_id skel_id,
                          const bool smooth_fac_from_iso,
                          Vec3_cu* d_output_vertices,
                          const float* d_base_potential,
                          Vec3_cu* d_gradient,
                          float* d_smooth_factors_iso,
                          float* d_smooth_factors,
                          int* d_vert_to_fit,
                          const int nb_vert_to_fit,
                          const unsigned short nb_iter,
                          const float gradient_threshold,
                          const float step_length,
                          const bool potential_pit,
                          EAnimesh::Vert_state *d_vert_state,
                          const float smooth_strength,
                          const int slope,
                          const bool raphson);


/*
 *
 * // TODO: to be deleted
__global__
void fill_grid_with_fngf(int bone_id,
                         float3 steps,
                         int grid_res,
                         Point_cu org,
                         Transfo transfo,
                         Device::Array<float4> d_out_grid);
*/

/// Compute on GPU the normals of the mesh using the normal at each face
void compute_normals(const int* tri,
                     Device::Array<Mesh::PrimIdxVertices> piv,
                     int nb_tri,
                     const Vec3_cu* vertices,
                     Device::Array<Vec3_cu> unpacked_normals,
                     int unpack_factor,
                     Vec3_cu* out_normals);

/// Tangential relaxation of the vertices. Each vertex is expressed with the
/// mean value coordinates (mvc) of its neighborhood. While animating we try
/// to move back the vertices to their old position with the mvc.
/// (N.B mvc are barycentric coordinates computed in the tangent plane of the
/// vertex, the plane can be defined either by the vertex's normal or
/// implicit gradient)
void conservative_smooth(Vec3_cu* d_vertices,
                         Vec3_cu* d_tmp_vertices,
                         Vec3_cu* d_normals,
                         const DA_int& d_edge_list,
                         const DA_int& d_edge_list_offsets,
                         const DA_float& d_edge_mvc,
                         const int* d_vert_to_fit,
                         int nb_vert_to_fit,
                         float strength,
                         int nb_iter,
                         const float* smooth_fac,
                         bool use_smooth_fac);

/// A basic laplacian smooth which move the vertices between its position
/// and the barycenter of its neighborhoods
/// @param factor sets for each vertex a weight between [0 1] which define
/// the smoothing strenght
/// @param use_smooth_factors do we use the array "factor" for smoothing
/// @param strength smoothing force when "use_smooth_factors"==false
void laplacian_smooth(Vec3_cu* d_vertices,
                      Vec3_cu* d_tmp_vertices,
                      DA_int d_edge_list,
                      DA_int d_edge_list_offsets,
                      const float* factors,
                      bool use_smooth_factors,
                      float strength,
                      int nb_iter,
                      int nb_min_neighbours);

/// A better laplacian smoothing algorithm which avoids shrinkage of the mesh
/// see article "Improved Laplacian Smoothing of Noisy Surface Meshes"
void hc_laplacian_smooth(const DA_Vec3_cu& d_original_vertices,
                         Vec3_cu* d_smoothed_vertices,
                         Vec3_cu* d_vector_correction,
                         Vec3_cu* d_tmp_vertices,
                         DA_int d_edge_list,
                         DA_int d_edge_list_offsets,
                         const float* factors,
                         bool use_smooth_factors,
                         float alpha,
                         float beta,
                         int nb_iter,
                         int nb_min_neighbours);


/// Basic diffusion of values on the mesh. For each vertex i we compute :
/// new_val(i) = val(i) * (1. - strength) + strength / sum( val(neighborhoods(i)) )
/// @param d_values diffused values computed in place
/// @param d_values_buffer an allocated buffer of the same size as 'd_values'
/// @param strenght is the strenght of the diffusion
/// @param nb_iter number of iteration to apply the diffusion. Avoid odd numbers
/// wich will imply a recopy of the array d_values, prefer even numbers
void diffuse_values(float* d_values,
                    float* d_values_buffer,
                    DA_int d_edge_list,
                    DA_int d_edge_list_offsets,
                    float strength,
                    int nb_iter);

/// Copy d_vertices_in of size n in d_vertices_out
template< class T >
__global__
void copy_arrays(const T* d_in, T* d_out, int n)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < n)  d_out[p] = d_in[p];
}

/// Fill the array with its the subscript index at each element
__global__
void fill_index(DA_int array);

/// @param d_in_vertices vertices to be smoothed
/// @param d_in_normals normals associated to the array d_in_vertices
/// @param d_out_vector Correction vector to aply to vertices in the final stage
/// of the smoothing
/// @param factors smoothing strength at each vertices
/// @param n number of vertices
__global__
void tangential_smooth_kernel_first_pass(const Vec3_cu* in_vertices,
                                         const Vec3_cu* in_normals,
                                         Vec3_cu* out_vector,
                                         const int* edge_list,
                                         const int* edge_list_offsets,
                                         const float* factors,
                                         bool use_smooth_factors,
                                         float strength,
                                         int nb_min_neighbours,
                                         int n);

__global__
void tangential_smooth_kernel_final_pass(const Vec3_cu* in_vertices,
                                         const Vec3_cu* in_vector,
                                         Vec3_cu* out_vertices,
                                         int n);



}// END Animesh_kers NAMESPACE =================================================

#endif // ANIMESH_KERS_HPP_
