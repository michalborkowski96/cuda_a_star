#ifndef DEVICE_HASHMAP_H
#define DEVICE_HASHMAP_H

#include "helpers.h"
#include "consts.h"

#include "reduction.h"
#include "functional_types.h"

namespace DeviceHashmapHelper {
	template<typename U, uint16_t bucket_depth>
	struct HashBucket {
		U values[bucket_depth];
		uint32_t total_used;
	};

	__device__ uint32_t get_bucket_id(uint32_t hash, uint32_t bucket_count) {
		return (hash * ((uint64_t) bucket_count)) >> 32;
	}


}

template<typename T, typename U, uint16_t bucket_depth, uint16_t minimum_free>
struct DeviceHashmap {
	DeviceHashmapHelper::HashBucket<U, bucket_depth>* buckets;

	__device__ DeviceHashmapHelper::HashBucket<U, bucket_depth>* get_buckets_ptr(){
		return (DeviceHashmapHelper::HashBucket<U, bucket_depth>*) (((size_t) buckets) & -2UL);
	}

	__device__ void mark(){
		buckets = (DeviceHashmapHelper::HashBucket<U, bucket_depth>*) (((size_t) buckets) | 1UL);
	}

	__device__ void unmark(){
		buckets = (DeviceHashmapHelper::HashBucket<U, bucket_depth>*) (((size_t) buckets) & -2UL);
	}

	__device__ bool marked() const {
		return ((size_t) buckets) & 1UL;
	}

	__host__ __device__ DeviceHashmap() = delete;

	__device__ DeviceHashmapHelper::HashBucket<U, bucket_depth>* get(const T& key, uint32_t size) {
		return get_buckets_ptr() + DeviceHashmapHelper::get_bucket_id(murmur3<T>(key), size);
	}

	__device__ void erase(const T& key, const U& value, uint32_t size) {
		uint32_t index = DeviceHashmapHelper::get_bucket_id(murmur3<T>(key), size);
		DeviceHashmapHelper::HashBucket<U, bucket_depth>& bucket = get_buckets_ptr()[index];
		int32_t j =  bucket.total_used - 1;
		while(j >= 0 && bucket.values[j] == value) {
			--j;
		}
		for(int32_t i = 0; i < j; ++i) {
			if(bucket.values[i] == value) {
				bucket.values[i] = bucket.values[j];
				--j;
				while(j > i && bucket.values[j] == value) {
					--j;
				}
			}
		}
		bucket.total_used = j + 1;
	}

	__device__ void add(const T& key, const U& value, uint32_t size) {
		uint32_t index = DeviceHashmapHelper::get_bucket_id(murmur3<T>(key), size);
		DeviceHashmapHelper::HashBucket<U, bucket_depth>* bucket = get_buckets_ptr() + index;
		bucket->values[bucket->total_used++] = value;
		if((bucket_depth - bucket->total_used) < minimum_free) {
			mark();
		}
	}

	__device__ void concurrent_add(const T& key, const U& value, uint32_t size) {
		uint32_t index = DeviceHashmapHelper::get_bucket_id(murmur3<T>(key), size);
		DeviceHashmapHelper::HashBucket<U, bucket_depth>* bucket = get_buckets_ptr() + index;
		uint32_t pos = atomicAdd(&(bucket->total_used), 1);
		if(pos >= size) {
			return;
		}
		bucket->values[pos] = value;
		if((bucket_depth - (pos + 1)) < minimum_free) {
			mark();
		}
	}
};

namespace DeviceHashmapHelper {
	template<typename T, typename U, uint16_t bucket_depth, uint16_t minimum_free>
	__global__ void zero_total_used(DeviceHashmap<T, U, bucket_depth, minimum_free>* hashmaps, uint32_t count, uint32_t size){
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= count * size) {
			return;
		}
		hashmaps[index / size].buckets[index % size].total_used = 0;
	}

	template<typename T, typename U, uint16_t bucket_depth, uint16_t minimum_free>
	__global__ void unmark(DeviceHashmap<T, U, bucket_depth, minimum_free>* hashmaps, uint32_t count){
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= count) {
			return;
		}
		hashmaps[index].unmark();
	}

	template<typename T, typename U, uint16_t bucket_depth, uint16_t minimum_free>
	__device__ bool is_marked(const DeviceHashmap<T, U, bucket_depth, minimum_free>& hashmap){
		return hashmap.marked();
	}
}

template<typename T, typename U, uint16_t bucket_depth, uint16_t minimum_free, typename std::enable_if_t<minimum_free < bucket_depth, int*> = nullptr>
class DeviceHashmapHost {
	DeviceHashmap<T, U, bucket_depth, minimum_free>* hashmaps_host;
	DeviceHashmap<T, U, bucket_depth, minimum_free>* hashmaps_device;
	__host__ __device__ DeviceHashmapHost() = delete;

	__host__ void unmark(){
		DeviceHashmapHelper::unmark<<<get_block_count(count), BLOCK_SIZE>>>(hashmaps_device, count);
	}
public:
	const uint32_t count;
	const uint32_t size;
	__host__ DeviceHashmapHost(uint32_t count, uint32_t size) : count(count), size(size) {
		hashmaps_host = (DeviceHashmap<T, U, bucket_depth, minimum_free>*) malloc(sizeof(DeviceHashmap<T, U, bucket_depth, minimum_free>) * count);
		cuda_check(cudaMalloc(&hashmaps_device, sizeof(DeviceHashmap<T, U, bucket_depth, minimum_free>) * count));
		for(uint32_t i = 0; i < count; ++i) {
			cuda_check(cudaMalloc(&(hashmaps_host[i].buckets), sizeof(DeviceHashmapHelper::HashBucket<U, bucket_depth>) * size));
		}
		cuda_check(cudaMemcpy(hashmaps_device, hashmaps_host, sizeof(DeviceHashmap<T, U, bucket_depth, minimum_free>) * count, cudaMemcpyHostToDevice));
		DeviceHashmapHelper::zero_total_used<<<get_block_count(size * count), BLOCK_SIZE>>>(hashmaps_device, count, size);
	}

	__host__ ~DeviceHashmapHost() {
		unmark();
		for(uint32_t i = 0; i < count; ++i) {
			cuda_check(cudaFree(hashmaps_host[i].buckets));
		}
		cuda_check(cudaFree(hashmaps_device));
		free(hashmaps_host);
	}

	__host__ DeviceHashmap<T, U, bucket_depth, minimum_free>* get_device_array() {
		return hashmaps_device;
	}

	__host__ bool full() const {
		return reduction_map<DeviceHashmap<T, U, bucket_depth, minimum_free>, bool, DeviceHashmapHelper::is_marked<T, U, bucket_depth, minimum_free>, bool_or<bool>>(hashmaps_device, count);
	}
};

#endif
