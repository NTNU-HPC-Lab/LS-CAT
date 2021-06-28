
#include "cuda_mparticles.cuh"
#include "cuda_bits.h"

// ----------------------------------------------------------------------
// check_in_patch_unordered_slow

template<typename BS>
bool cuda_mparticles<BS>::check_in_patch_unordered_slow()
{
  auto n_prts_by_patch = this->sizeByPatch();

  uint off = 0;
  for (int p = 0; p < this->n_patches(); p++) {
    for (int n = 0; n < n_prts_by_patch[p]; n++) {
      int bidx = this->blockIndex(this->storage.load(off + n), p);
      if (!(bidx >= 0 && bidx <= this->n_blocks)) return false;
    }
    off += n_prts_by_patch[p];
  }

  if (!(off == this->n_prts)) return false;
  // printf("PASS: cuda_mparticles_check_in_patch_unordered_slow()\n");
  return true;
}

// ----------------------------------------------------------------------
// check_bix_id_unordered_slow
//
// checks that block indices are correct,
// id is just enumerating particles

template<typename BS>
bool cuda_mparticles<BS>::check_bidx_id_unordered_slow()
{
  auto n_prts_by_patch = this->sizeByPatch();

  uint off = 0;
  for (int p = 0; p < this->n_patches(); p++) {
    for (int n = 0; n < n_prts_by_patch[p]; n++) {
      int bidx = this->blockIndex(this->storage.load(off + n), p);
      if (!(bidx == this->by_block_.d_idx[off+n])) return false;
      if (!(off+n == this->by_block_.d_id[off+n])) return false;
    }
    off += n_prts_by_patch[p];
  }

  if (!(off == this->n_prts)) return false;
  // printf("PASS: cuda_mparticles_check_bidx_id_unordered_slow()\n");
  return true;
}

// ----------------------------------------------------------------------
// check_ordered

template<typename BS>
bool cuda_mparticles<BS>::check_ordered()
{
  thrust::host_vector<float4> h_xi4(this->storage.xi4);
  thrust::host_vector<uint> h_off(this->by_block_.d_off);
  thrust::host_vector<uint> h_id(this->by_block_.d_id);

  //printf("check_ordered: need_reorder %s\n", need_reorder ? "true" : "false");

  uint off = 0;
  for (int b = 0; b < this->n_blocks; b++) {
    int p = b / this->n_blocks_per_patch;
    uint off_b = h_off[b], off_e = h_off[b+1];
    if (!(off_e >= off_b)) return false;
    //printf("check_ordered: block %d: %d -> %d (patch %d)\n", b, off_b, off_e, p);
    if (!(off_b == off)) return false;
    for (int n = h_off[b]; n < h_off[b+1]; n++) {
      float4 xi4;
      if (need_reorder) {
	xi4 = h_xi4[h_id[n]];
      } else {
	xi4 = h_xi4[n];
      }
      uint bidx = this->blockIndex(xi4, p);
      //printf("check_ordered: bidx %d\n", bidx);
      if (b != bidx) {
	printf("check_ordered: b %d bidx %d n %d p %d xi4 %g %g %g\n",
	       b, bidx, n, p, xi4.x, xi4.y, xi4.z);
	Int3 bpos = this->blockPosition(&xi4.x);
	printf("block_pos %d %d\n", bpos[1], bpos[2]);
      }
      if (!(b == bidx)) return false;
    }
    off += off_e - off_b;
  }
  if (!(off == this->n_prts)) return false;
  // printf("PASS: cuda_mparticles_check_ordered:\n");
  return true;
}

// ----------------------------------------------------------------------
// check_bidx_after_push

template<typename BS>
bool cuda_mparticles<BS>::check_bidx_after_push()
{
  bool ok = true;

  thrust::host_vector<uint> h_off(this->by_block_.d_off);
  thrust::host_vector<uint> h_bidx(this->by_block_.d_idx);
  thrust::host_vector<float4> h_xi4(this->storage.xi4);
  
  for (int p = 0; p < this->n_patches(); p++) {
    int begin = h_off[p * this->n_blocks_per_patch];
    int end = h_off[(p+1) * this->n_blocks_per_patch];
    for (int n = begin; n < end; n++) {
      float4 xi4 = h_xi4[n];
      int bidx = h_bidx[n];
      int bidx2 = this->blockIndex(xi4, p);
      if (bidx2 < 0) bidx2 = this->n_blocks;
      if (bidx != bidx2) {
	mprintf("check_bidx: n %d: xi4 %g %g %g bidx %d/%d\n", n, xi4.x, xi4.y, xi4.z,
		bidx, bidx2);
	ok = false;
      }
    }
  }
  return ok;
}

