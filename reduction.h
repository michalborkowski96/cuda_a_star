#ifndef REDUCTION_H
#define REDUCTION_H

#include "consts.h"
#include "functional_types.h"
#include "helpers.h"

template<typename T>
__host__ __device__ T identity(const T& a) {
	return a;
}

template<typename T, typename U, UnOp<T, U> un_op, BinOp<U> bin_op>
__global__ void dev_reduction_map(const T* source, const uint32_t size, U* target) {
	__shared__ char cdata[BLOCK_SIZE * sizeof(U)];
	U* data = (U*) cdata;
	uint32_t tid = threadIdx.x;
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if(blockIdx.x == gridDim.x - 1 && (size % BLOCK_SIZE)) {
		uint32_t last_block_size = size % BLOCK_SIZE;
		if(index < size) {
			data[tid] = un_op(source[index]);
		}
		__syncthreads();
		for(uint32_t i = blockDim.x >> 1; i > 0; i >>= 1) {
			if(tid < i && tid + i < last_block_size) {
				data[tid] = bin_op(data[tid], data[tid + i]);
			}
			__syncthreads();
		}
	} else {
		data[tid] = un_op(source[index]);
		__syncthreads();
		for(uint32_t i = blockDim.x >> 1; i > 0; i >>= 1) {
			if(tid < i) {
				data[tid] = bin_op(data[tid], data[tid + i]);
			}
			__syncthreads();
		}
	}
	if(tid == 0) {
		target[blockIdx.x] = data[0];
	}
}

__host__ void* reduction_get_tmp_bytes(uint32_t size) {
	static void* tmp = nullptr;
	static uint32_t tmp_size = 0;
	if (tmp_size < size) {
		tmp_size = size;
		if(tmp) {
			cuda_check(cudaFree(tmp));
		}
		cuda_check(cudaMalloc(&tmp, size));
	}
	return tmp;
}

template<typename U>
__host__ U* reduction_get_tmp(uint32_t size) {
	return (U*) reduction_get_tmp_bytes(size * sizeof(U));
}

template<typename T, typename U, UnOp<T, U> un_op, BinOp<U> bin_op>
__host__ U reduction_map(const T* source, uint32_t size) {
	uint32_t blocks = get_block_count(size);
	U* tmp = reduction_get_tmp<U>(blocks);
	dev_reduction_map<T, U, un_op, bin_op><<<blocks, BLOCK_SIZE>>>(source, size, tmp);
	size = blocks;
	blocks = get_block_count(size);
	while(size != 1) {
		dev_reduction_map<U, U, identity<U>, bin_op><<<blocks, BLOCK_SIZE>>>(tmp, size, tmp);
		size = blocks;
		blocks = get_block_count(size);
	}
	char ret[sizeof(U)];
	cuda_check(cudaMemcpy(ret, tmp, sizeof(U), cudaMemcpyDeviceToHost));
	return *((U*) ret);
}

template<typename T, BinOp<T> bin_op>
__host__ T reduction(const T* source, uint32_t size) {
	return reduction_map<T, T, identity<T>, bin_op>(source, size);
}

#endif
