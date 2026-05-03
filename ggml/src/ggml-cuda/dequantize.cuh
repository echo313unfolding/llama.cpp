#include "common.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// HXQ affine g128: 8-bit indices, scale+offset per group of 128
// W[i] = qs[i] * scale + offset
static __device__ __forceinline__ void dequantize_hxq_affine_g128(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_hxq_affine_g128 * x = (const block_hxq_affine_g128 *) vx;

    const float scale  = x[ib].scale;
    const float offset = x[ib].offset;

    v.x = x[ib].qs[iqs + 0] * scale + offset;
    v.y = x[ib].qs[iqs + 1] * scale + offset;
}

// HXQ affine 6-bit: 6-bit indices packed 4 per 3 bytes, scale+offset per group of 128
// W[i] = idx6[i] * scale + offset
// iqs indexes pairs of elements (0, 2, 4, ... 126)
static __device__ __forceinline__ void dequantize_hxq_affine_6(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_hxq_affine_6 * x = (const block_hxq_affine_6 *) vx;

    const float scale  = x[ib].scale;
    const float offset = x[ib].offset;

    // iqs is the element pair index (0..126 stepping by 2)
    // Each group of 4 elements occupies 3 bytes
    // Element j is in group (j/4), at position (j%4) within the group
    const int j0 = iqs;
    const int j1 = iqs + 1;

    // Unpack element j0
    const int group0    = j0 / 4;
    const int pos0      = j0 % 4;
    const int byte_off0 = group0 * 3;
    const uint8_t b0_0  = x[ib].qs[byte_off0 + 0];
    const uint8_t b0_1  = x[ib].qs[byte_off0 + 1];
    const uint8_t b0_2  = x[ib].qs[byte_off0 + 2];

    int idx0;
    switch (pos0) {
        case 0: idx0 =  b0_0       & 0x3F; break;
        case 1: idx0 = ((b0_0 >> 6) | (b0_1 << 2)) & 0x3F; break;
        case 2: idx0 = ((b0_1 >> 4) | (b0_2 << 4)) & 0x3F; break;
        default: idx0 = b0_2 >> 2; break;
    }

    // Unpack element j1
    const int group1    = j1 / 4;
    const int pos1      = j1 % 4;
    const int byte_off1 = group1 * 3;
    const uint8_t b1_0  = x[ib].qs[byte_off1 + 0];
    const uint8_t b1_1  = x[ib].qs[byte_off1 + 1];
    const uint8_t b1_2  = x[ib].qs[byte_off1 + 2];

    int idx1;
    switch (pos1) {
        case 0: idx1 =  b1_0       & 0x3F; break;
        case 1: idx1 = ((b1_0 >> 6) | (b1_1 << 2)) & 0x3F; break;
        case 2: idx1 = ((b1_1 >> 4) | (b1_2 << 4)) & 0x3F; break;
        default: idx1 = b1_2 >> 2; break;
    }

    v.x = idx0 * scale + offset;
    v.y = idx1 * scale + offset;
}
