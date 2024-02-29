
#include "performancecounters/benchmarker.h"
#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <vector>

#define DEBUG_YMM(reg) printf(#reg "=%u,%u,%u,%u,%u,%u,%u,%u\n", (uint32_t) _mm256_extract_epi32(reg, 0), (uint32_t) _mm256_extract_epi32(reg, 1), (uint32_t) _mm256_extract_epi32(reg, 2), (uint32_t) _mm256_extract_epi32(reg, 3), (uint32_t) _mm256_extract_epi32(reg, 4), (uint32_t) _mm256_extract_epi32(reg, 5), (uint32_t) _mm256_extract_epi32(reg, 6), (uint32_t) _mm256_extract_epi32(reg, 7))

uint32_t karprabin_hash(const char *data, size_t len, uint32_t B) {
    uint32_t hash = 0;
    for (size_t i = 0; i < len; i++) {
        hash = hash * B + data[i];
    }
    return hash;
}

__attribute__ ((noinline))
size_t karprabin_naive(const char *data, size_t len, size_t N, uint32_t B, uint32_t target) {
    size_t counter = 0;
    for (size_t i = 0; i < len - N; i++) {
        uint32_t hash = 0;
        for (size_t j = 0; j < N; j++) {
            hash = hash * B + data[i + j];
        }
        if (hash == target) {
            counter++;
        }
    }
    return counter;
}

__attribute__ ((noinline))
size_t karprabin_rolling(const char *data, size_t len, size_t N, uint32_t B, uint32_t target) {
    size_t counter = 0;
    uint32_t BtoN = 1;
    for (size_t i = 0; i < N; i++) {
        BtoN *= B;
    }
    uint32_t hash = 0;
    for (size_t i = 0; i < N; i++) {
        hash = hash * B + data[i];
    }
    if (hash == target) {
        counter++;
    }
    for (size_t i = N; i < len; i++) {
        hash = hash * B + data[i] - BtoN * data[i - N];
        // karprabin_hash(data+i-N+1, N, B) == hash
        if (hash == target) {
            counter++;
        }
    }
    return counter;
}

__attribute__ ((noinline))
size_t sum_rolling(const char *data, size_t len, size_t N, uint32_t B, uint32_t target) {
    size_t counter = 0;

    uint32_t hash = 0;
    for (size_t i = 0; i < N; i++) {
        hash = hash + data[i];
    }
    if (hash == target) {
        counter++;
    }
    for (size_t i = N; i < len; i++) {
        hash = hash + data[i] - data[i - N];
        if (hash == target) {
            counter++;
        }
    }
    return counter;
}


__attribute__ ((noinline))
size_t karprabin_rolling4(const char *data, size_t len, size_t N, uint32_t B, uint32_t target) {
    size_t counter = 0;
    uint32_t BtoN = 1;
    for (size_t i = 0; i < N; i++) {
        BtoN *= B;
    }
    uint32_t hash = 0;
    for (size_t i = 0; i < N; i++) {
        hash = hash * B + data[i];
    }
    if (hash == target) {
        counter++;
    }
    size_t i = N;
    for (; i + 3 < len; i += 4) {
        hash = hash * B + data[i] - BtoN * data[i - N];
        if (hash == target) {
            counter++;
        }
        hash = hash * B + data[i + 1] - BtoN * data[i + 1 - N];
        if (hash == target) {
            counter++;
        }
        hash = hash * B + data[i + 2] - BtoN * data[i + 2 - N];
        if (hash == target) {
            counter++;
        }
        hash = hash * B + data[i + 3] - BtoN * data[i + 3 - N];
        if (hash == target) {
            counter++;
        }
    }

    for (; i < len; i++) {
        hash = hash * B + data[i] - BtoN * data[i - N];
        // karprabin_hash(data+i-N+1, N, B) == hash
        if (hash == target) {
            counter++;
        }
    }
    return counter;
}

__attribute__ ((noinline))
size_t karprabin_rolling4_split_2(const char *data, size_t len, size_t N, uint32_t B, uint32_t target) {
    size_t counter = 0;
    uint32_t BtoN = 1;
    for (size_t i = 0; i < N; i++) {
        BtoN *= B;
    }
    uint32_t hash0 = 0;
    uint32_t hash1 = 0;

    size_t end0 = len / 2;
    size_t end1 = len;

    size_t i0 = 0;
    // Need to process an extra window
    size_t i1 = end0 - N;

    for (size_t i = 0; i < N; i++) {
        hash0 = hash0 * B + data[i0++];
        hash1 = hash1 * B + data[i1++];
    }
    if (hash0 == target) {
        counter++;
    }
    if (hash1 == target) {
        counter++;
    }

    while (i1 + 3 < end1) {
        hash0 = hash0 * B + data[i0] - BtoN * data[i0 - N];
        hash1 = hash1 * B + data[i1] - BtoN * data[i1 - N];
        if (hash0 == target) {
            counter++;
        }
        if (hash1 == target) {
            counter++;
        }

        hash0 = hash0 * B + data[i0 + 1] - BtoN * data[i0 + 1 - N];
        hash1 = hash1 * B + data[i1 + 1] - BtoN * data[i1 + 1 - N];
        if (hash0 == target) {
            counter++;
        }
        if (hash1 == target) {
            counter++;
        }

        hash0 = hash0 * B + data[i0 + 2] - BtoN * data[i0 + 2 - N];
        hash1 = hash1 * B + data[i1 + 2] - BtoN * data[i1 + 2 - N];
        if (hash0 == target) {
            counter++;
        }
        if (hash1 == target) {
            counter++;
        }

        hash0 = hash0 * B + data[i0 + 3] - BtoN * data[i0 + 3 - N];
        hash1 = hash1 * B + data[i1 + 3] - BtoN * data[i1 + 3 - N];
        if (hash0 == target) {
            counter++;
        }
        if (hash1 == target) {
            counter++;
        }

        i0 += 4;
        i1 += 4;
    }

    for (; i0 < end0; i0++) {
        hash0 = hash0 * B + data[i0] - BtoN * data[i0 - N];
        if (hash0 == target) {
            counter++;
        }
    }
    for (; i1 < end1; i1++) {
        hash1 = hash1 * B + data[i1] - BtoN * data[i1 - N];
        if (hash1 == target) {
            counter++;
        }
    }
    return counter;
}

__attribute__ ((noinline))
size_t karprabin_rolling4_split_4(const char *data, size_t len, size_t N, uint32_t B, uint32_t target) {
    size_t counter = 0;
    uint32_t BtoN = 1;
    for (size_t i = 0; i < N; i++) {
        BtoN *= B;
    }
    uint32_t hash0 = 0;
    uint32_t hash1 = 0;
    uint32_t hash2 = 0;
    uint32_t hash3 = 0;

    size_t end0 = len / 4;
    size_t end1 = 2 * (len / 4);
    size_t end2 = 3 * (len / 4);
    size_t end3 = len;

    size_t i0 = 0;
    size_t i1 = end0 - N;
    size_t i2 = end1 - N;
    size_t i3 = end2 - N;

    for (size_t i = 0; i < N; i++) {
        hash0 = hash0 * B + data[i0++];
        hash1 = hash1 * B + data[i1++];
        hash2 = hash2 * B + data[i2++];
        hash3 = hash3 * B + data[i3++];
    }
    if (hash0 == target) {
        counter++;
    }
    if (hash1 == target) {
        counter++;
    }
    if (hash2 == target) {
        counter++;
    }
    if (hash3 == target) {
        counter++;
    }

    while (i3 + 3 < end3) {
        hash0 = hash0 * B + data[i0] - BtoN * data[i0 - N];
        hash1 = hash1 * B + data[i1] - BtoN * data[i1 - N];
        hash2 = hash2 * B + data[i2] - BtoN * data[i2 - N];
        hash3 = hash3 * B + data[i3] - BtoN * data[i3 - N];
        if (hash0 == target) {
            counter++;
        }
        if (hash1 == target) {
            counter++;
        }
        if (hash2 == target) {
            counter++;
        }
        if (hash3 == target) {
            counter++;
        }

        hash0 = hash0 * B + data[i0 + 1] - BtoN * data[i0 + 1 - N];
        hash1 = hash1 * B + data[i1 + 1] - BtoN * data[i1 + 1 - N];
        hash2 = hash2 * B + data[i2 + 1] - BtoN * data[i2 + 1 - N];
        hash3 = hash3 * B + data[i3 + 1] - BtoN * data[i3 + 1 - N];
        if (hash0 == target) {
            counter++;
        }
        if (hash1 == target) {
            counter++;
        }
        if (hash2 == target) {
            counter++;
        }
        if (hash3 == target) {
            counter++;
        }

        hash0 = hash0 * B + data[i0 + 2] - BtoN * data[i0 + 2 - N];
        hash1 = hash1 * B + data[i1 + 2] - BtoN * data[i1 + 2 - N];
        hash2 = hash2 * B + data[i2 + 2] - BtoN * data[i2 + 2 - N];
        hash3 = hash3 * B + data[i3 + 2] - BtoN * data[i3 + 2 - N];
        if (hash0 == target) {
            counter++;
        }
        if (hash1 == target) {
            counter++;
        }
        if (hash2 == target) {
            counter++;
        }
        if (hash3 == target) {
            counter++;
        }

        hash0 = hash0 * B + data[i0 + 3] - BtoN * data[i0 + 3 - N];
        hash1 = hash1 * B + data[i1 + 3] - BtoN * data[i1 + 3 - N];
        hash2 = hash2 * B + data[i2 + 3] - BtoN * data[i2 + 3 - N];
        hash3 = hash3 * B + data[i3 + 3] - BtoN * data[i3 + 3 - N];
        if (hash0 == target) {
            counter++;
        }
        if (hash1 == target) {
            counter++;
        }
        if (hash2 == target) {
            counter++;
        }
        if (hash3 == target) {
            counter++;
        }

        i0 += 4;
        i1 += 4;
        i2 += 4;
        i3 += 4;
    }

    for (; i0 < end0; i0++) {
        hash0 = hash0 * B + data[i0] - BtoN * data[i0 - N];
        if (hash0 == target) {
            counter++;
        }
    }
    for (; i1 < end1; i1++) {
        hash1 = hash1 * B + data[i1] - BtoN * data[i1 - N];
        if (hash1 == target) {
            counter++;
        }
    }
    for (; i2 < end2; i2++) {
        hash2 = hash2 * B + data[i2] - BtoN * data[i2 - N];
        if (hash2 == target) {
            counter++;
        }
    }
    for (; i3 < end3; i3++) {
        hash3 = hash3 * B + data[i3] - BtoN * data[i3 - N];
        if (hash3 == target) {
            counter++;
        }
    }

    return counter;
}

__attribute__ ((noinline))
size_t karprabin_rolling4_leaping_2x4(const char *data, size_t len, size_t N, uint32_t B, uint32_t target) {
    size_t counter = 0;
    uint32_t BtoN = 1;
    for (size_t i = 0; i < N; i++) {
        BtoN *= B;
    }

    // Must be divisible by 8
    size_t subblock_size = 16 * N;
    size_t block_size = 2 * subblock_size;

    // Inclusive index through which all hashes have been computed
    size_t last_end = 0;
    // Last hash that has been computed
    uint32_t last_hash = 0;

    while (last_end + block_size + N < len) {
        // Initialize hashes
        uint32_t hash0 = 0;
        uint32_t hash1 = 0;

        size_t i0;
        size_t i1;
        if (last_end < N) {
            i0 = 0;
            i1 = subblock_size;
        } else {
            i0 = last_end - N;
            i1 = last_end + subblock_size - N;
        }

        for (size_t i = 0; i < N; i++) {
            hash0 = hash0 * B + data[i0++];
            hash1 = hash1 * B + data[i1++];
        }
        if (hash0 == target && last_end == 0) {
            counter++;
        }
        if (hash1 == target && last_end == 0) {
            counter++;
        }

//        printf("STARTING AT %ld,%ld\n", i0, i1);

        for (size_t i = 0; i < subblock_size; i+=4) {
            hash0 = hash0 * B + data[i0] - BtoN * data[i0 - N];
            hash1 = hash1 * B + data[i1] - BtoN * data[i1 - N];
            if (hash0 == target) {
                counter++;
            }
            if (hash1 == target) {
                counter++;
            }

            hash0 = hash0 * B + data[i0 + 1] - BtoN * data[i0 + 1 - N];
            hash1 = hash1 * B + data[i1 + 1] - BtoN * data[i1 + 1 - N];
            if (hash0 == target) {
                counter++;
            }
            if (hash1 == target) {
                counter++;
            }

            hash0 = hash0 * B + data[i0 + 2] - BtoN * data[i0 + 2 - N];
            hash1 = hash1 * B + data[i1 + 2] - BtoN * data[i1 + 2 - N];
            if (hash0 == target) {
                counter++;
            }
            if (hash1 == target) {
                counter++;
            }

            hash0 = hash0 * B + data[i0 + 3] - BtoN * data[i0 + 3 - N];
            hash1 = hash1 * B + data[i1 + 3] - BtoN * data[i1 + 3 - N];
            if (hash0 == target) {
                counter++;
            }
            if (hash1 == target) {
                counter++;
            }

            i0 += 4;
            i1 += 4;
        }

        last_end = i1;
        last_hash = hash1;
//        printf("THROUGH=%ld\n", last_end);
//        printf("COUNT=%ld\n", counter);
    }

    uint32_t hash = last_hash;
    for (size_t i = last_end + 1; i < len; i++) {
        hash = hash * B + data[i] - BtoN * data[i - N];
        if (hash == target) {
            counter++;
        }
    }

    return counter;
}

static inline __m256i constant_ymm(uint32_t x) {
    return _mm256_set_epi32(x, x, x, x, x, x, x, x);
}

// Byte extractors

static inline __m256i byte0(__m256i bytes) {
    return _mm256_srai_epi32(_mm256_slli_epi32(bytes, 24), 24);
}

static inline __m256i byte1(__m256i bytes) {
    return _mm256_srai_epi32(_mm256_slli_epi32(bytes, 16), 24);
}

static inline __m256i byte2(__m256i bytes) {
    return _mm256_srai_epi32(_mm256_slli_epi32(bytes, 8), 24);
}

static inline __m256i byte3(__m256i bytes) {
    return _mm256_srai_epi32(bytes, 24);
}

__attribute__ ((noinline))
size_t karprabin_rolling4_leaping_8x4_avx2(const char *data, size_t len, size_t N, uint32_t B, uint32_t target) {
    uint32_t BtoN = 1;
    for (size_t i = 0; i < N; i++) {
        BtoN *= B;
    }

    size_t counter = 0;

    size_t subblock_size = 4 * N;
    size_t block_size = 8 * subblock_size;

    __m256i targets = constant_ymm(target);
    __m256i bs = constant_ymm(B);
    __m256i bns = constant_ymm(BtoN);
    __m256i offsets = _mm256_set_epi32(
        7 * subblock_size,
        6 * subblock_size,
        5 * subblock_size,
        4 * subblock_size,
        3 * subblock_size,
        2 * subblock_size,
        subblock_size,
        0
    );

    __m256i hashes = _mm256_setzero_si256();
    const char* block_start = data;
    while ((block_start - data) + block_size + N < len) {
        // Initialize hashes
        hashes = _mm256_setzero_si256();

        for (size_t i = 0; i < N; i++) {
            __m256i as = _mm256_srai_epi32(
                _mm256_slli_epi32(
                    _mm256_i32gather_epi32((const int *) (block_start + i), offsets, 1),
                    24
                ),
                24
            );
            hashes = _mm256_add_epi32(
                _mm256_mullo_epi32(hashes, bs),
                as
            );
        }

        uint32_t first = _mm256_extract_epi32(hashes, 0);
        if (first == target) {
            counter++;
        }

        for (size_t i = 0; i < subblock_size; i+=4) {
            // Values to be added in
            __m256i as = _mm256_i32gather_epi32(
                (const int*) (block_start + i + N),
                offsets,
                1
            );
            // Values to be dropped off
            __m256i ans = _mm256_i32gather_epi32(
                (const int*) (block_start + i),
                offsets,
                1
            );

            // Value 0
            hashes = _mm256_sub_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(hashes, bs),
                    byte0(as)
                ),
                _mm256_mullo_epi32(byte0(ans), bns)
            );
            counter += __builtin_popcount(_mm256_cmpeq_epi32_mask(hashes, targets));

            // Value 1
            hashes = _mm256_sub_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(hashes, bs),
                    byte1(as)
                ),
                _mm256_mullo_epi32(byte1(ans), bns)
            );
            counter += __builtin_popcount(_mm256_cmpeq_epi32_mask(hashes, targets));

            // Value 3
            hashes = _mm256_sub_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(hashes, bs),
                    byte2(as)
                ),
                _mm256_mullo_epi32(byte2(ans), bns)
            );
            counter += __builtin_popcount(_mm256_cmpeq_epi32_mask(hashes, targets));

            // Value 4
            hashes = _mm256_sub_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(hashes, bs),
                    byte3(as)
                ),
                _mm256_mullo_epi32(byte3(ans), bns)
            );
            counter += __builtin_popcount(_mm256_cmpeq_epi32_mask(hashes, targets));
        }

        block_start += block_size + 1;
    }

    // Deal with what's left over
    size_t last_end = (block_start - data) + N - 1;
    uint32_t hash = _mm256_extract_epi32(hashes, 7);
    for (size_t i = last_end; i < len; i++) {
        hash = hash * B + data[i] - BtoN * data[i - N];
        if (hash == target) {
            counter++;
        }
    }

    return counter;
}


__attribute__ ((noinline))
size_t karprabin_rolling4_leaping_8x4_avx2_2(const char *data, size_t len, size_t N, uint32_t B, uint32_t target) {
    uint32_t BtoN = 1;
    for (size_t i = 0; i < N; i++) {
        BtoN *= B;
    }

    size_t counter = 0;

    size_t subblock_size = 4 * N;
    size_t block_size = 8 * subblock_size;

    __m256i static_targets = constant_ymm(target);

    __m256i static_bs = constant_ymm(B);
    __m256i static_bs4 = constant_ymm(B*B*B*B);

    __m256i offsets = _mm256_set_epi32(
        7 * subblock_size,
        6 * subblock_size,
        5 * subblock_size,
        4 * subblock_size,
        3 * subblock_size,
        2 * subblock_size,
        subblock_size,
        0
    );

    const char* block_start = data;
    while ((block_start - data) + block_size + N < len) {
        // Initialize hashes
        __m256i hashes = _mm256_setzero_si256();

        // All ones
        __m256i bs1 = _mm256_srli_epi32(_mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()), 31);
        __m256i bs2;
        __m256i bs3;
        __m256i bs4;

        for (size_t i = 0; i < N; i++) {
            __m256i as = _mm256_srai_epi32(
                _mm256_slli_epi32(
                    _mm256_i32gather_epi32((const int *) (block_start + i), offsets, 1),
                    24
                ),
                24
            );
            hashes = _mm256_add_epi32(
                hashes,
                _mm256_mullo_epi32(as, bs1)
            );
            bs1 = _mm256_mullo_epi32(bs1, static_bs);
        }
        bs2 = _mm256_mullo_epi32(bs1, static_bs);
        bs3 = _mm256_mullo_epi32(bs2, static_bs);
        bs4 = _mm256_mullo_epi32(bs3, static_bs);

        __m256i bns1 = _mm256_srli_epi32(_mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()), 31);
        __m256i bns2 = _mm256_mullo_epi32(bns1, static_bs);
        __m256i bns3 = _mm256_mullo_epi32(bns2, static_bs);
        __m256i bns4 = _mm256_mullo_epi32(bns3, static_bs);

        __m256i targets1 = _mm256_mullo_epi32(static_targets, static_bs);
        __m256i targets2 = _mm256_mullo_epi32(targets1, static_bs);
        __m256i targets3 = _mm256_mullo_epi32(targets2, static_bs);
        __m256i targets4 = _mm256_mullo_epi32(targets3, static_bs);

        uint32_t first = _mm256_extract_epi32(hashes, 0);
        if (first == target) {
            counter++;
        }

        for (size_t i = 0; i < subblock_size; i+=4) {
            __m256i as = _mm256_i32gather_epi32(
                (const int*) (block_start + i + N),
                offsets,
                1
            );
            __m256i ans = _mm256_i32gather_epi32(
                (const int*) (block_start + i),
                offsets,
                1
            );

            // Value 0
            hashes = _mm256_sub_epi32(
                _mm256_add_epi32(
                    hashes,
                    _mm256_mullo_epi32(byte0(as), bs1)
                ),
                _mm256_mullo_epi32(byte0(ans), bns1)
            );
            counter += __builtin_popcount(_mm256_cmpeq_epi32_mask(hashes, targets1));

            // Value 1
            hashes = _mm256_sub_epi32(
                _mm256_add_epi32(
                    hashes,
                    _mm256_mullo_epi32(byte1(as), bs2)
                ),
                _mm256_mullo_epi32(byte1(ans), bns2)
            );
            counter += __builtin_popcount(_mm256_cmpeq_epi32_mask(hashes, targets2));

            // Value 2
            hashes = _mm256_sub_epi32(
                _mm256_add_epi32(
                    hashes,
                    _mm256_mullo_epi32(byte2(as), bs3)
                ),
                _mm256_mullo_epi32(byte2(ans), bns3)
            );
            counter += __builtin_popcount(_mm256_cmpeq_epi32_mask(hashes, targets3));

            // Value 3
            hashes = _mm256_sub_epi32(
                _mm256_add_epi32(
                    hashes,
                    _mm256_mullo_epi32(byte3(as), bs4)
                ),
                _mm256_mullo_epi32(byte3(ans), bns4)
            );
            counter += __builtin_popcount(_mm256_cmpeq_epi32_mask(hashes, targets4));

            // Update all factors
            bs1 = _mm256_mullo_epi32(bs1, static_bs4);
            bs2 = _mm256_mullo_epi32(bs2, static_bs4);
            bs3 = _mm256_mullo_epi32(bs3, static_bs4);
            bs4 = _mm256_mullo_epi32(bs4, static_bs4);
            bns1 = _mm256_mullo_epi32(bns1, static_bs4);
            bns2 = _mm256_mullo_epi32(bns2, static_bs4);
            bns3 = _mm256_mullo_epi32(bns3, static_bs4);
            bns4 = _mm256_mullo_epi32(bns4, static_bs4);
            targets1 = _mm256_mullo_epi32(targets1, static_bs4);
            targets2 = _mm256_mullo_epi32(targets2, static_bs4);
            targets3 = _mm256_mullo_epi32(targets3, static_bs4);
            targets4 = _mm256_mullo_epi32(targets4, static_bs4);
        }

        block_start += block_size + 1;
    }

    size_t last_end = (block_start - data) + N - 1;
    counter += karprabin_rolling(
        data + last_end - N + 1,
        len - last_end + N,
        N,
        B,
        target
    );

    return counter;
}

__attribute__ ((noinline))
size_t karprabin_rolling4_leaping_16x2_avx512(const char *data, size_t len, size_t N, uint32_t B, uint32_t target) {
    uint32_t BtoN = 1;
    for (size_t i = 0; i < N; i++) {
        BtoN *= B;
    }

    size_t counter = 0;

    size_t subblock_size = 16 * N;
    size_t block_size = 2 * subblock_size;

    __m512i static_targets = _mm512_broadcastd_epi32(_mm_insert_epi32(_mm_setzero_si128(), target, 0));
//    printf("TARGET=%d\n", (int32_t) target);
    __m512i static_bs = _mm512_broadcastd_epi32(_mm_insert_epi32(_mm_setzero_si128(), B, 0));
    __m512i static_bs4 = _mm512_mullo_epi32(
        static_bs,
        _mm512_mullo_epi32(
            static_bs,
            _mm512_mullo_epi32(static_bs, static_bs)
        )
    );

    __m512i offsets = _mm512_set_epi32(
        15 * subblock_size,
        14 * subblock_size,
        13 * subblock_size,
        12 * subblock_size,
        11 * subblock_size,
        10 * subblock_size,
        9 * subblock_size,
        8 * subblock_size,
        7 * subblock_size,
        6 * subblock_size,
        5 * subblock_size,
        4 * subblock_size,
        3 * subblock_size,
        2 * subblock_size,
        1 * subblock_size,
        0
    );
//    offsets = _mm512_insert_epi32(offsets, _mm_insert_epi32(, 0), 0);
//    offsets = _mm512_insert_epi32(offsets, subblock_size, 1);
//    offsets = _mm512_insert_epi32(offsets, 2 * subblock_size, 2);
//    offsets = _mm512_insert_epi32(offsets, 3 * subblock_size, 3);
//    offsets = _mm512_insert_epi32(offsets, 4 * subblock_size, 4);
//    offsets = _mm512_insert_epi32(offsets, 5 * subblock_size, 5);
//    offsets = _mm512_insert_epi32(offsets, 6 * subblock_size, 6);
//    offsets = _mm512_insert_epi32(offsets, 7 * subblock_size, 7);
//    offsets = _mm512_insert_epi32(offsets, 8 * subblock_size, 8);
//    offsets = _mm512_insert_epi32(offsets, 9 * subblock_size, 9);
//    offsets = _mm512_insert_epi32(offsets, 10 * subblock_size, 10);
//    offsets = _mm512_insert_epi32(offsets, 11 * subblock_size, 11);
//    offsets = _mm512_insert_epi32(offsets, 12 * subblock_size, 12);
//    offsets = _mm512_insert_epi32(offsets, 13 * subblock_size, 13);
//    offsets = _mm512_insert_epi32(offsets, 14 * subblock_size, 14);
//    offsets = _mm512_insert_epi32(offsets, 15 * subblock_size, 15);
    __m512i ones = _mm512_set_epi32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
    __m512i hashes = _mm512_setzero_si512();
    const char* block_start = data;
    while ((block_start - data) + block_size + N < len) {
        // Initialize hashes
        hashes = _mm512_setzero_si512();
//        __m256i targets = static_targets;
        // All ones
        __m512i bs1 = ones;
        __m512i bs2;
        __m512i bs3;
        __m512i bs4;

        for (size_t i = 0; i < N; i++) {
            __m512i as = _mm512_srai_epi32(
                _mm512_slli_epi32(
                    _mm512_i32gather_epi32(offsets, block_start + i, 1),
                    24
                ),
                24
            );
            hashes = _mm512_add_epi32(
                hashes,
                _mm512_mullo_epi32(as, bs1)
            );
            bs1 = _mm512_mullo_epi32(bs1, static_bs);
//            DEBUG_YMM(hashes);
        }
        bs2 = _mm512_mullo_epi32(bs1, static_bs);
        bs3 = _mm512_mullo_epi32(bs2, static_bs);
        bs4 = _mm512_mullo_epi32(bs3, static_bs);

        __m512i bns1 = ones;
        __m512i bns2 = _mm512_mullo_epi32(bns1, static_bs);
        __m512i bns3 = _mm512_mullo_epi32(bns2, static_bs);
        __m512i bns4 = _mm512_mullo_epi32(bns3, static_bs);

        __m512i targets1 = _mm512_mullo_epi32(static_targets, static_bs);
//        DEBUG_YMM(static_targets);
        __m512i targets2 = _mm512_mullo_epi32(targets1, static_bs);
        __m512i targets3 = _mm512_mullo_epi32(targets2, static_bs);
        __m512i targets4 = _mm512_mullo_epi32(targets3, static_bs);

        uint32_t first = _mm_extract_epi32(_mm512_extracti32x4_epi32(hashes, 0), 0);
        if (first == target) {
            counter++;
        }
//        printf("\n");
//        printf("\n");

        for (size_t i = 0; i < subblock_size; i+=4) {
            __m512i as = _mm512_i32gather_epi32(
                offsets,
                block_start + i + N,
                1
            );
            __m512i ans = _mm512_i32gather_epi32(
                offsets,
                block_start + i,
                1
            );

            // Value 0
            hashes = _mm512_sub_epi32(
                _mm512_add_epi32(
                    hashes,
                    _mm512_mullo_epi32(_mm512_srai_epi32(_mm512_slli_epi32(as, 24), 24), bs1)
                ),
                _mm512_mullo_epi32(_mm512_srai_epi32(_mm512_slli_epi32(ans, 24), 24), bns1)
            );
            counter += __builtin_popcount(_mm512_cmpeq_epi32_mask(hashes, targets1));

            // Value 1
            hashes = _mm512_sub_epi32(
                _mm512_add_epi32(
                    hashes,
                    _mm512_mullo_epi32(_mm512_srai_epi32(_mm512_slli_epi32(as, 16), 24), bs2)
                ),
                _mm512_mullo_epi32(_mm512_srai_epi32(_mm512_slli_epi32(ans, 16), 24), bns2)
            );
            counter += __builtin_popcount(_mm512_cmpeq_epi32_mask(hashes, targets2));

            // Value 3
            hashes = _mm512_sub_epi32(
                _mm512_add_epi32(
                    hashes,
                    _mm512_mullo_epi32(_mm512_srai_epi32(_mm512_slli_epi32(as, 8), 24), bs3)
                ),
                _mm512_mullo_epi32(_mm512_srai_epi32(_mm512_slli_epi32(ans, 8), 24), bns3)
            );
            counter += __builtin_popcount(_mm512_cmpeq_epi32_mask(hashes, targets3));

            // Value 4
            hashes = _mm512_sub_epi32(
                _mm512_add_epi32(
                    hashes,
                    _mm512_mullo_epi32(_mm512_srai_epi32(as, 24), bs4)
                ),
                _mm512_mullo_epi32(_mm512_srai_epi32(ans, 24), bns4)
            );
            counter += __builtin_popcount(_mm512_cmpeq_epi32_mask(hashes, targets4));

            // Update all factors
            bs1 = _mm512_mullo_epi32(bs1, static_bs4);
            bs2 = _mm512_mullo_epi32(bs2, static_bs4);
            bs3 = _mm512_mullo_epi32(bs3, static_bs4);
            bs4 = _mm512_mullo_epi32(bs4, static_bs4);
            bns1 = _mm512_mullo_epi32(bns1, static_bs4);
            bns2 = _mm512_mullo_epi32(bns2, static_bs4);
            bns3 = _mm512_mullo_epi32(bns3, static_bs4);
            bns4 = _mm512_mullo_epi32(bns4, static_bs4);
            targets1 = _mm512_mullo_epi32(targets1, static_bs4);
            targets2 = _mm512_mullo_epi32(targets2, static_bs4);
            targets3 = _mm512_mullo_epi32(targets3, static_bs4);
            targets4 = _mm512_mullo_epi32(targets4, static_bs4);
        }

//        DEBUG_YMM(counts);
        block_start += block_size + 1;
    }

    size_t last_end = (block_start - data) + N - 1;
    counter += karprabin_rolling(
        data + last_end - N + 1,
        len - last_end + N,
        N,
        B,
        target
    );

    return counter;
}


void pretty_print(size_t volume, size_t bytes, std::string name,
                  event_aggregate agg) {
    printf("%-40s : ", name.c_str());
    printf(" %5.2f GB/s ", bytes / agg.fastest_elapsed_ns());
    printf(" %5.1f Ma/s ", volume * 1000.0 / agg.fastest_elapsed_ns());
    printf(" %5.2f ns/d ", agg.fastest_elapsed_ns() / volume);
    if (collector.has_events()) {
        printf(" %5.2f GHz ", agg.cycles() / agg.elapsed_ns());
        printf(" %5.2f c/d ", agg.fastest_cycles() / volume);
        printf(" %5.2f i/d ", agg.fastest_instructions() / volume);
        printf(" %5.2f c/b ", agg.fastest_cycles() / bytes);
        printf(" %5.2f i/b ", agg.fastest_instructions() / bytes);
        printf(" %5.2f i/c ", agg.fastest_instructions() / agg.fastest_cycles());
    }
    printf("\n");
}
void init(float *v, size_t N) {
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dis(0, 1);
    for (size_t i = 0; i < N; i++) {
        v[i] = dis(gen);
    }
}
int main(int argc, char **argv) {

    printf("please be patient, this will take a few seconds...\n");
    const size_t N = 100 * 1000 * 1000;
    std::unique_ptr<char[]> data(new char[N]);
    for (size_t i = 0; i < N; i++) {
        data.get()[i] = i % 256;
    }

    for (size_t i = N - 10000; i < N; i++) {
        data.get()[i] = 1;
    }

    for (size_t i = 0; i < 10000; i++) {
        data.get()[i] = 1;
    }

    uint32_t target = 3902431073;

    printf("Reference %ld\n", karprabin_rolling(data.get(), N, 75, 31, target));
    printf("Check %ld\n", karprabin_rolling4_split_2(data.get(), N, 75, 31, target));
    printf("Check %ld\n", karprabin_rolling4_split_4(data.get(), N, 75, 31, target));
    printf("Check %ld\n", karprabin_rolling4_leaping_2x4(data.get(), N, 75, 31, target));
    printf("Check %ld\n", karprabin_rolling4_leaping_8x4_avx2(data.get(), N, 75, 31, target));
    printf("Check %ld\n", karprabin_rolling4_leaping_8x4_avx2_2(data.get(), N, 75, 31, target));
    printf("Check %ld\n", karprabin_rolling4_leaping_16x2_avx512(data.get(), N, 75, 31, target));

    size_t volume = N;
    volatile size_t counter = 0;
    for (size_t window = 64; window <= 4096; window *= 2) {
        printf("window = %zu\n", window);
        pretty_print(1, volume, "karprabin_rolling4", bench([&data, &counter, &window, &target]() {
            counter += karprabin_rolling4(data.get(), N, window, 31, target);
        }));
        pretty_print(1, volume, "karprabin_rolling4_split_2", bench([&data, &counter, &window, &target]() {
            counter += karprabin_rolling4_split_2(data.get(), N, window, 31, target);
        }));
        pretty_print(1, volume, "karprabin_rolling4_split_4", bench([&data, &counter, &window, &target]() {
            counter += karprabin_rolling4_split_4(data.get(), N, window, 31, target);
        }));
        pretty_print(1, volume, "karprabin_rolling4_leaping_2x4", bench([&data, &counter, &window, &target]() {
            counter += karprabin_rolling4_leaping_2x4(data.get(), N, window, 31, target);
        }));
        pretty_print(1, volume, "karprabin_rolling4_leaping_8x4_avx2", bench([&data, &counter, &window, &target]() {
            counter += karprabin_rolling4_leaping_8x4_avx2(data.get(), N, window, 31, target);
        }));
        pretty_print(1, volume, "karprabin_rolling4_leaping_8x4_avx2_2", bench([&data, &counter, &window, &target]() {
            counter += karprabin_rolling4_leaping_8x4_avx2_2(data.get(), N, window, 31, target);
        }));
        pretty_print(1, volume, "karprabin_rolling4_leaping_16x2_avx512", bench([&data, &counter, &window, &target]() {
            counter += karprabin_rolling4_leaping_8x4_avx2_2(data.get(), N, window, 31, target);
        }));
        pretty_print(1, volume, "karprabin_rolling", bench([&data, &counter, &window, &target]() {
            counter += karprabin_rolling(data.get(), N, window, 31, target);
        }));
    }
}
