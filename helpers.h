#ifndef HELPERS_H
#define HELPERS_H

#include "consts.h"

#include <type_traits>
#include <cstdint>
#include <stdexcept>

template<typename T, typename = std::enable_if_t<sizeof(T) % 4 == 0>>
__host__ __device__ uint32_t murmur3 (const T& key) {
	uint32_t hash = 0;

	const uint32_t* beginning = (const uint32_t*) &key;
	const uint32_t* end = (const uint32_t*) ((&key) + 1);

	for(const uint32_t* i = beginning; i != end; ++i) {
		uint32_t k = *i;
		k *= 0xcc9e2d51;
		k = (k << 15) | (k >> 17);
		k *= 0x1b873593;
		hash ^= k;
		hash = (hash << 13) | (hash >> 19);
		hash = hash * 5 + 0xe6546b64;
	}

	hash ^= sizeof(key);
	hash ^= hash >> 16;
	hash *= 0x85ebca6b;
	hash ^= hash >> 13;
	hash *= 0xc2b2ae35;
	hash ^= hash >> 16;

	return hash;
}

template <typename, template <typename...> typename>
struct is_instance : public std::false_type {};

template <typename T, template <typename...> typename U>
struct is_instance<U<T>, U> : public std::true_type {};

template <typename A, typename B>
struct is_same_kind : public std::is_same<std::remove_cv_t<std::remove_reference_t<A>>, std::remove_cv_t<std::remove_reference_t<B>>> {};

__host__ void cuda_check(cudaError_t err) {
	if(err != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorName(err));
	}
}

__host__ uint32_t get_block_count(uint32_t elements) {
	uint32_t blocks = elements / BLOCK_SIZE;
	if (elements % BLOCK_SIZE) {
		++blocks;
	}
	return blocks;
}

__host__ uint32_t get_nearest_power_of_two(uint32_t v){
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

template<typename T, typename U>
__host__ __device__ void swap_with(T* a, T* b) {
	U* ai = (U*) a;
	U* bi = (U*) b;
	U s;
	for(int32_t i = 0; i < sizeof(T) / sizeof(U); ++i) {
		s = *ai;
		*ai = *bi;
		*bi = s;
		++ai;
		++bi;
	}
}

template<typename T>
__host__ __device__ void swap(T* a, T* b) {
	if(sizeof(T) % sizeof(uint64_t) == 0) {
		swap_with<T, uint64_t>(a, b);
	} else if(sizeof(T) % sizeof(uint32_t) == 0) {
		swap_with<T, uint32_t>(a, b);
	} else if(sizeof(T) % sizeof(uint16_t) == 0) {
		swap_with<T, uint16_t>(a, b);
	} else {
		swap_with<T, unsigned char>(a, b);
	}
}

template<typename T, typename = std::enable_if_t<sizeof(T) % 4 == 0>>
__host__ __device__ bool mem_eq(const T* a, const T* b) {
	const uint32_t* ai = (const uint32_t*) a;
	const uint32_t* bi = (const uint32_t*) b;
	const uint32_t* aend = (const uint32_t*) (a + 1);
	while(ai != aend) {
		if(*ai != *bi) {
			return false;
		}
		++ai;
		++bi;
	}
	return true;
}

template<typename T, typename = std::enable_if_t<sizeof(T) % 4 == 0>>
__host__ __device__ bool mem_less(const T* a, const T* b) {
	const uint32_t* ai = (const uint32_t*) a;
	const uint32_t* bi = (const uint32_t*) b;
	const uint32_t* aend = (const uint32_t*) (a + 1);
	while(ai != aend) {
		if(*ai < *bi) {
			return true;
		}
		if(*ai > *bi) {
			return false;
		}
		++ai;
		++bi;
	}
	return false;
}

template<typename T>
__device__ T sum(const T& a, const T& b) {
	return a + b;
}

template<typename T>
__device__ T max(const T& a, const T& b) {
	return b < a ? a : b;
}

template<typename T>
__device__ T min(const T& a, const T& b) {
	return a < b ? a : b;
}

template<typename T>
__device__ T bool_or(const T& a, const T& b) {
	return a || b;
}

#endif
