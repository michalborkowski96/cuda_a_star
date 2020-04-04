#ifndef FANCY_HASHMAP_H
#define FANCY_HASHMAP_H

#define FANCY_HASHMAP_ENTRY_STATUS_EMPTY 0
#define FANCY_HASHMAP_ENTRY_STATUS_NEW 1
#define FANCY_HASHMAP_ENTRY_STATUS_REJECTED 2
#define FANCY_HASHMAP_ENTRY_STATUS_INSERTED 3
#define FANCY_HASHMAP_ENTRY_STATUS_UPDATED 4

#include "helpers.h"
#include "consts.h"
#include "reduction.h"
#include "functional_types.h"

#include <type_traits>

namespace FancyHashmapHelper {
	template<typename U, typename = std::enable_if_t<sizeof(int32_t) < sizeof(U)>>
	union HashBucketValue {
		U v;
		int32_t source_index;
	};

	template<typename T, typename U, uint16_t bucket_depth, typename = std::enable_if_t<bucket_depth % 4 == 0>, typename = std::enable_if_t<(sizeof(T) + sizeof(U)) % 4 == 0>>
	struct HashBucket {
		T keys[bucket_depth];
		HashBucketValue<U> values[bucket_depth];
		uint32_t old_total_used;
		uint32_t total_used;
	};

	template<typename T, typename U, uint16_t bucket_depth>
	__global__ void remove_duplicates(HashBucket<T, U, bucket_depth>** buckets, uint32_t bucket_count, const T* keys, const U* values, unsigned char* element_status) {
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= bucket_count) {
			return;
		}

		HashBucket<T, U, bucket_depth>* bucket = buckets[index];

		uint32_t total_used = bucket->total_used;
		uint32_t old_total_used = bucket->old_total_used;

		if(total_used > bucket_depth) {
			total_used = bucket_depth;
		}

		for(uint32_t i = 0; i < old_total_used; ++i) {
			int32_t j = old_total_used;
			for(; j < total_used; ++j) {
				int32_t jj = bucket->values[j].source_index;
				if(jj < 0) {
					continue;
				}
				if(bucket->keys[i] == keys[jj]) {
					if(bucket->values[i].v <= values[jj]) {
						element_status[jj] = FANCY_HASHMAP_ENTRY_STATUS_REJECTED;
						bucket->values[j].source_index = -1;
					} else {
						element_status[jj] = FANCY_HASHMAP_ENTRY_STATUS_UPDATED;
					}
				}
			}
		}

		uint32_t tt = total_used;

		total_used = old_total_used;

		for(uint32_t i = old_total_used; i < tt; ++i) {
			int32_t ii = bucket->values[i].source_index;
			if(ii >= 0) {
				if(i != total_used) {
					bucket->values[total_used].source_index = ii;
				}
				++total_used;
			}
		}

		bucket->total_used = total_used;
	}

	template<typename T, typename U, uint16_t bucket_depth>
	__global__ void copy_data(HashBucket<T, U, bucket_depth>** buckets, uint32_t bucket_count, const T* keys, U* values, const unsigned char* element_status) {
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= bucket_count) {
			return;
		}
		HashBucket<T, U, bucket_depth>* bucket = buckets[index];

		uint32_t old_total_used = bucket->old_total_used;
		uint32_t total_used = bucket->total_used;

		for(uint32_t j = old_total_used; j < total_used; ++j) {
			int32_t jj = bucket->values[j].source_index;
			bucket->keys[j] = keys[jj];
			bucket->values[j].v = values[jj];
			if(element_status[jj] == FANCY_HASHMAP_ENTRY_STATUS_UPDATED) {
				for(uint32_t i = 0; i < old_total_used; ++i) {
					if(bucket->keys[i] == bucket->keys[j]) {
						values[jj].set_old(bucket->values[i].v);
						break;
					}
				}
			}
		}
	}

	template<typename T, typename U, uint16_t bucket_depth>
	__global__ void densify(HashBucket<T, U, bucket_depth>** buckets, uint32_t bucket_count) {
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= bucket_count) {
			return;
		}
		HashBucket<T, U, bucket_depth>* bucket = buckets[index];

		uint32_t old_total_used = bucket->old_total_used;
		uint32_t tt = bucket->total_used;

		bool unused[bucket_depth] = {0};

		for(uint32_t i = 0; i < old_total_used; ++i) {
			for(uint32_t j = old_total_used; j < tt; ++j) {
				if(unused[j]) {
					continue;
				}
				if(bucket->keys[i] == bucket->keys[j]) {
					unused[j] = true;
					bucket->values[i].v = bucket->values[j].v;
					bucket->keys[i] = bucket->keys[j];
					break;
				}
			}
		}

		uint32_t total_used = old_total_used;
		for(uint32_t i = old_total_used; i < tt; ++i) {
			if(!unused[i]) {
				if(i != total_used) {
					bucket->keys[total_used] = bucket->keys[i];
					bucket->values[total_used].v = bucket->values[i].v;
				}
				++total_used;
			}
		}

		bucket->total_used = total_used;
		bucket->old_total_used = total_used;
	}

	__device__ uint32_t get_bucket_id(uint32_t hash, uint32_t bucket_count) {
		return (hash * ((uint64_t) bucket_count)) >> 32;
	}

	template<typename T, typename U, uint16_t bucket_depth>
	__global__ void try_insert(HashBucket<T, U, bucket_depth>** buckets, uint32_t bucket_count, const T* keys, unsigned char* element_status, uint32_t element_count) {
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= element_count || (element_status[index] != FANCY_HASHMAP_ENTRY_STATUS_NEW)) {
			return;
		}
		uint32_t bucket_index = get_bucket_id(murmur3<T>(keys[index]), bucket_count);
		uint32_t in_bucket_pos = atomicAdd(&(buckets[bucket_index]->total_used), 1);
		if(in_bucket_pos < bucket_depth) {
			buckets[bucket_index]->values[in_bucket_pos].source_index = index;
			element_status[index] = FANCY_HASHMAP_ENTRY_STATUS_INSERTED;
		}
	}

	template<typename T, typename U, uint16_t bucket_depth>
	__global__ void rehash(HashBucket<T, U, bucket_depth>** buckets, HashBucket<T, U, bucket_depth>** new_buckets, uint32_t old_bucket_count){
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= old_bucket_count * bucket_depth) {
			return;
		}
		uint32_t bucket_index = index / bucket_depth;
		uint32_t in_bucket_index = index % bucket_depth;
		if(in_bucket_index >= buckets[bucket_index]->total_used) {
			return;
		}
		uint32_t new_bucket_index = get_bucket_id(murmur3<T>(buckets[bucket_index]->keys[in_bucket_index]), old_bucket_count << 1);
		if((new_bucket_index & 1) == 0) {
			return;
		}
		HashBucket<T, U, bucket_depth>* new_bucket = new_buckets[new_bucket_index];
		uint32_t new_in_bucket_index = atomicAdd(&(new_bucket->total_used), 1);
		new_bucket->keys[new_in_bucket_index] = buckets[bucket_index]->keys[in_bucket_index];
		new_bucket->values[new_in_bucket_index] = buckets[bucket_index]->values[in_bucket_index];
	}

	template<typename T, typename U, uint16_t bucket_depth>
	__global__ void move_even(HashBucket<T, U, bucket_depth>** buckets, HashBucket<T, U, bucket_depth>** new_buckets, uint32_t old_bucket_count){
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= old_bucket_count) {
			return;
		}
		new_buckets[index << 1] = buckets[index];
	}

	template<typename T, typename U, uint16_t bucket_depth>
	__global__ void remove_moved(HashBucket<T, U, bucket_depth>** buckets, uint32_t new_bucket_count){
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= (new_bucket_count >> 1)) {
			return;
		}

		HashBucket<T, U, bucket_depth> cached_bucket = *(buckets[index]);

		uint32_t total_used = 0;
		for(uint32_t i = 0; i < cached_bucket.total_used; ++i) {
			if((get_bucket_id(murmur3<T>(cached_bucket.keys[i]), new_bucket_count) & 1) == 0) {
				if(i != total_used) {
					cached_bucket.keys[total_used] = cached_bucket.keys[i];
					cached_bucket.values[total_used].v = cached_bucket.values[i].v;
				}
				++total_used;
			}
		}

		cached_bucket.total_used = total_used;

		*(buckets[index]) = cached_bucket;
	}

	template<typename T, typename U, uint16_t bucket_depth>
	__global__ void copy_total_used_to_old(HashBucket<T, U, bucket_depth>** buckets, uint32_t bucket_count){
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= bucket_count) {
			return;
		}
		buckets[index]->old_total_used = buckets[index]->total_used;
	}

	template<typename T, typename U, uint16_t bucket_depth>
	__global__ void wipe(HashBucket<T, U, bucket_depth>** buckets, uint32_t bucket_count){
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= bucket_count) {
			return;
		}
		buckets[index]->old_total_used = 0;
		buckets[index]->total_used = 0;
	}

	template<typename T, typename U, uint16_t bucket_depth>
	__global__ void wipe_every_odd(HashBucket<T, U, bucket_depth>** buckets, uint32_t bucket_count){
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		index *= 2;
		index += 1;
		if(index >= bucket_count) {
			return;
		}
		buckets[index]->old_total_used = 0;
		buckets[index]->total_used = 0;
	}

	template<typename T, typename U, uint16_t bucket_depth, typename = std::enable_if_t<(1U < bucket_depth)>>
	struct FancyHashmapBase {
	protected:
		uint32_t bucket_count;
		HashBucket<T, U, bucket_depth>** buckets;
		FancyHashmapBase() = delete;
		FancyHashmapBase(uint32_t bucket_count) : bucket_count(bucket_count) {
			if(bucket_count < 2) {
				this->bucket_count = 2;
			}
		}
	};

	template<typename T>
	__device__ T is_element_new(const unsigned char& b) {
		return b == FANCY_HASHMAP_ENTRY_STATUS_NEW;
	}
}

template<typename T, typename U, uint16_t bucket_depth>
class FancyHashmapHost : public FancyHashmapHelper::FancyHashmapBase<T, U, bucket_depth> {
	using FancyHashmapHelper::FancyHashmapBase<T, U, bucket_depth>::bucket_count;
	using FancyHashmapHelper::FancyHashmapBase<T, U, bucket_depth>::buckets;
	FancyHashmapHelper::HashBucket<T, U, bucket_depth>** buckets_host;

	__host__ void expand() {
		FancyHashmapHelper::HashBucket<T, U, bucket_depth>** new_buckets_host;
		FancyHashmapHelper::HashBucket<T, U, bucket_depth>** new_buckets_device;

		uint32_t new_bucket_count = bucket_count << 1;
		new_buckets_host = new FancyHashmapHelper::HashBucket<T, U, bucket_depth>*[new_bucket_count];;
		for(uint32_t i = 1; i < new_bucket_count; i += 2) {
			cuda_check(cudaMalloc(new_buckets_host + i, sizeof(FancyHashmapHelper::HashBucket<T, U, bucket_depth>)));
		}
		cuda_check(cudaMalloc(&new_buckets_device, new_bucket_count * sizeof(FancyHashmapHelper::HashBucket<T, U, bucket_depth>*)));
		cuda_check(cudaMemcpy(new_buckets_device, new_buckets_host, new_bucket_count * sizeof(FancyHashmapHelper::HashBucket<T, U, bucket_depth>*), cudaMemcpyHostToDevice));
		FancyHashmapHelper::wipe_every_odd<<<get_block_count(bucket_count), BLOCK_SIZE>>>(new_buckets_device, new_bucket_count);
		FancyHashmapHelper::rehash<T, U, bucket_depth><<<get_block_count(bucket_depth * bucket_count), BLOCK_SIZE>>>(buckets, new_buckets_device, bucket_count);
		FancyHashmapHelper::remove_moved<T, U, bucket_depth><<<get_block_count(bucket_count), BLOCK_SIZE>>>(buckets, new_bucket_count);
		FancyHashmapHelper::move_even<T, U, bucket_depth><<<get_block_count(bucket_count), BLOCK_SIZE>>>(buckets, new_buckets_device, bucket_count);
		cuda_check(cudaMemcpy(new_buckets_host, new_buckets_device, new_bucket_count * sizeof(FancyHashmapHelper::HashBucket<T, U, bucket_depth>*), cudaMemcpyDeviceToHost));
		FancyHashmapHelper::copy_total_used_to_old<T, U, bucket_depth><<<get_block_count(new_bucket_count), BLOCK_SIZE>>>(new_buckets_device, new_bucket_count);
		delete[] buckets_host;
		cuda_check(cudaFree(buckets));
		buckets = new_buckets_device;
		bucket_count = new_bucket_count;
		buckets_host = new_buckets_host;
	}

	__host__ void insert_without_expanding_iteration(const T* keys, U* values, unsigned char* element_status, uint32_t element_count) {
		FancyHashmapHelper::try_insert<T, U, bucket_depth><<<get_block_count(element_count), BLOCK_SIZE>>>(buckets, bucket_count, keys, element_status, element_count);
		FancyHashmapHelper::remove_duplicates<T, U, bucket_depth><<<get_block_count(bucket_count), BLOCK_SIZE>>>(buckets, bucket_count, keys, values, element_status);
		FancyHashmapHelper::copy_data<T, U, bucket_depth><<<get_block_count(bucket_count), BLOCK_SIZE>>>(buckets, bucket_count, keys, values, element_status);
		FancyHashmapHelper::densify<T, U, bucket_depth><<<get_block_count(bucket_count), BLOCK_SIZE>>>(buckets, bucket_count);
	}

	__host__ uint32_t insert_without_expanding(const T* keys, U* values, unsigned char* element_status, uint32_t element_count) {
		insert_without_expanding_iteration(keys, values, element_status, element_count);

		uint32_t left_elements = reduction_map<unsigned char, uint32_t, FancyHashmapHelper::is_element_new<uint32_t>, sum<uint32_t>>(element_status, element_count);
		uint32_t old_left_elements = 0;
		while(old_left_elements != left_elements) {
			old_left_elements = left_elements;
			insert_without_expanding_iteration(keys, values, element_status, element_count);
			left_elements = reduction_map<unsigned char, uint32_t, FancyHashmapHelper::is_element_new<uint32_t>, sum<uint32_t>>(element_status, element_count);
		}
		return left_elements;
	}
public:
	__host__ __device__ FancyHashmapHost() = delete;
	__host__ __device__ FancyHashmapHost(const FancyHashmapHost&) = delete;
	__host__ FancyHashmapHost(uint32_t initial_size) : FancyHashmapHelper::FancyHashmapBase<T, U, bucket_depth>(initial_size) {
		buckets_host = new FancyHashmapHelper::HashBucket<T, U, bucket_depth>*[bucket_count];
		cuda_check(cudaMalloc(&buckets, bucket_count * sizeof(FancyHashmapHelper::HashBucket<T, U, bucket_depth>*)));
		for(uint32_t i = 0; i < bucket_count; ++i) {
			cuda_check(cudaMalloc(buckets_host + i, sizeof(FancyHashmapHelper::HashBucket<T, U, bucket_depth>)));
		}
		cuda_check(cudaMemcpy(buckets, buckets_host, bucket_count * sizeof(FancyHashmapHelper::HashBucket<T, U, bucket_depth>*), cudaMemcpyHostToDevice));
		FancyHashmapHelper::wipe<<<get_block_count(bucket_count), BLOCK_SIZE>>>(buckets, bucket_count);
	}

	__host__ void insert_unique_data(const T* keys, U* values, unsigned char* element_status, uint32_t element_count) {
		while(insert_without_expanding(keys, values, element_status, element_count)) {
			expand();
		}
	}

	__host__ ~FancyHashmapHost() {
		for(uint32_t i = 0; i < bucket_count; ++i) {
			cuda_check(cudaFree(buckets_host[i]));
		}
		cuda_check(cudaFree(buckets));
		delete[] buckets_host;
	}

	__host__ uint32_t get_bucket_count() const {
		return bucket_count;
	}
};

template<typename T, typename U, uint16_t bucket_depth>
class FancyHashmapDevice : public FancyHashmapHelper::FancyHashmapBase<T, U, bucket_depth> {
	using FancyHashmapHelper::FancyHashmapBase<T, U, bucket_depth>::bucket_count;
	using FancyHashmapHelper::FancyHashmapBase<T, U, bucket_depth>::buckets;
public:
	__host__ __device__ FancyHashmapDevice() = delete;
	__host__ __device__ FancyHashmapDevice(const FancyHashmapHelper::FancyHashmapBase<T, U, bucket_depth>& o) : FancyHashmapHelper::FancyHashmapBase<T, U, bucket_depth>(o) {}
	__device__ U* get(const T& key) {
		FancyHashmapHelper::HashBucket<T, U, bucket_depth>* bucket = buckets[FancyHashmapHelper::get_bucket_id(murmur3<T>(key), bucket_count)];
		for(uint32_t i = 0; i < bucket->total_used; ++i) {
			if(bucket->keys[i] == key) {
				return &(bucket->values[i].v);
			}
		}
		return nullptr;
	}
};

#endif
