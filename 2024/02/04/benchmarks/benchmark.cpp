
#include "performancecounters/benchmarker.h"
#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <vector>

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
    const size_t N = 100 * 1024 * 1024;
    std::unique_ptr<char[]> data(new char[N]);
    for (size_t i = 0; i < N; i++) {
        data.get()[i] = i % 256;
    }

    for (size_t i = N - 100000; i < N; i++) {
        data.get()[i] = 0;
    }
    printf("Reference %ld\n", karprabin_rolling(data.get(), N, 75, 31, 0));
    printf("Check %ld\n", karprabin_rolling4_split_2(data.get(), N, 75, 31, 0));
    printf("Check %ld\n", karprabin_rolling4_split_4(data.get(), N, 75, 31, 0));

    size_t volume = N;
    volatile size_t counter = 0;
    for (size_t window = 64; window <= 4096; window *= 2) {
        printf("window = %zu\n", window);
        pretty_print(1, volume, "karprabin_rolling4", bench([&data, &counter, &window]() {
            counter += karprabin_rolling4(data.get(), N, window, 31, 0);
        }));
        pretty_print(1, volume, "karprabin_rolling4_split_2", bench([&data, &counter, &window]() {
            counter += karprabin_rolling4_split_2(data.get(), N, window, 31, 0);
        }));
        pretty_print(1, volume, "karprabin_rolling4_split_4", bench([&data, &counter, &window]() {
            counter += karprabin_rolling4_split_4(data.get(), N, window, 31, 0);
        }));
        pretty_print(1, volume, "karprabin_rolling", bench([&data, &counter, &window]() {
            counter += karprabin_rolling(data.get(), N, window, 31, 0);
        }));
//        pretty_print(1, volume, "karprabin_naive", bench([&data, &counter, &window]() {
//            counter += karprabin_naive(data.get(), N, window, 31, 0);
//        }));
    }
}
