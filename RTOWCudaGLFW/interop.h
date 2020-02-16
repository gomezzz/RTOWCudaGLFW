#pragma once

#include <cuda_runtime.h>
#include <stdbool.h>

//Adapted from https://gist.github.com/allanmac/4ff11985c3562830989f by allanmac

struct cuda_glfw_interop* cuda_glfw_interop_create(const bool multi_gpu, const int fbo_count);

void cuda_glfw_interop_destroy(struct cuda_glfw_interop* const interop);

void cuda_glfw_interop_size_set(struct cuda_glfw_interop* const interop, const int width, const int height);

void cuda_glfw_interop_size_get(struct cuda_glfw_interop* const interop, int* const width, int* const height);

void cuda_glfw_interop_array_map(struct cuda_glfw_interop* const interop);

cudaArray_const_t cuda_glfw_interop_array_get(struct cuda_glfw_interop* const interop);

int cuda_glfw_interop_index_get(struct cuda_glfw_interop* const interop);

void cuda_glfw_interop_swap(struct cuda_glfw_interop* const interop);

void cuda_glfw_interop_clear(struct cuda_glfw_interop* const interop);

void cuda_glfw_interop_blit(struct cuda_glfw_interop* const interop);

