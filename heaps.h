#ifndef HEAPS_H
#define HEAPS_H

#include "reduction.h"
#include "helpers.h"
#include "functional_types.h"
#include "device_hashmap.h"

#include <utility>
#include <stdexcept>

#define MEMORY_CHUNK_SIZE 16

template<typename T, typename U>
struct HeapEntry {
	T key;
	U value;
	__device__ bool operator<(const HeapEntry<T, U>& o) const {
		return value < o.value;
	}
	__host__ __device__ HeapEntry() = delete;
	__device__ HeapEntry(const T& key, const U& value) : key(key), value(value) {}
};

namespace HeapsHelper {
	template<typename T, typename U>
	struct HeapsBase {
		__host__ __device__ uint32_t get_heaps_count() const {
			return heaps_count;
		}
	protected:
		static const uint32_t DEFAULT_MAX_ELEMENTS = 1024; //must be a power of two
		static const uint32_t HASHMAP_BUCKET_DEPTH = 64;
		uint32_t heaps_count;
		uint32_t* elements;
		uint32_t max_elements;
		HeapEntry<T, U>** data_device;
	};

	template<typename T, typename U>
	__device__ HeapEntry<T, U>* get_heap(HeapEntry<T, U>** data_device, uint32_t i, uint32_t max_elements){
		return data_device[i / MEMORY_CHUNK_SIZE] + (i % MEMORY_CHUNK_SIZE) * max_elements;
	}

	template<typename T, typename U>
	__global__ void get_min_elements(U* target, const uint32_t* elements, HeapEntry<T, U>** data_device, const U default_element, uint32_t count, uint32_t max_elements) {
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= count) {
			return;
		}
		target[index] = elements[index] ? get_heap(data_device, index, max_elements)[0].value : default_element;
	}

	template<typename T, typename U>
	__global__ void copy_data(uint32_t max_elements, uint32_t heaps_count, HeapEntry<T, U>** data_device, HeapEntry<T, U>** new_data_device) {
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= max_elements * heaps_count) {
			return;
		}
		uint32_t heap_index = index / max_elements;
		uint32_t element_index = index % max_elements;
		get_heap(new_data_device, heap_index, max_elements << 1)[element_index] = get_heap(data_device, heap_index, max_elements)[element_index];
	}

	template<typename T, typename U, uint16_t bucket_depth, uint16_t minimum_free>
	__global__ void insert_into_hashmaps(uint32_t* elements, HeapEntry<T, U>** data, uint32_t heaps_count, uint32_t max_elements, DeviceHashmap<T, uint32_t, bucket_depth, minimum_free>* hashmaps, uint32_t hashmap_size) {
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= heaps_count * max_elements) {
			return;
		}
		uint32_t heap_index = index / max_elements;
		uint32_t in_heap_index = index % max_elements;
		if(in_heap_index >= elements[heap_index]) {
			return;
		}
		hashmaps[heap_index].concurrent_add(get_heap(data, heap_index, max_elements)[in_heap_index].key, in_heap_index, hashmap_size);
	}
}

template<typename T, typename U, uint16_t minimum_free>
struct HeapsHost;

template<typename T, typename U, uint16_t minimum_free>
class HeapsDevice : public HeapsHelper::HeapsBase<T, U> {
	using HeapsHelper::HeapsBase<T, U>::HASHMAP_BUCKET_DEPTH;
	using HeapsHelper::HeapsBase<T, U>::heaps_count;
	using HeapsHelper::HeapsBase<T, U>::elements;
	using HeapsHelper::HeapsBase<T, U>::data_device;
	using HeapsHelper::HeapsBase<T, U>::max_elements;
	DeviceHashmap<T, uint32_t, HeapsHelper::HeapsBase<T, U>::HASHMAP_BUCKET_DEPTH, minimum_free>* hashmaps;
	const uint32_t hashmaps_size;

	__device__ HeapEntry<T, U> remove(uint32_t i, uint32_t pos) {
		HeapEntry<T, U>* data = HeapsHelper::get_heap(data_device, i, max_elements);
		HeapEntry<T, U> result = data[pos];
		uint32_t els = elements[i] - 1;
		data[pos] = data[els];
		elements[i] = els;
		uint32_t index = pos, left_child, right_child;
		hashmaps[i].erase(result.key, pos, hashmaps_size);
		hashmaps[i].erase(data[pos].key, els, hashmaps_size);
		if(pos != els) {
			hashmaps[i].add(data[pos].key, pos, hashmaps_size);
		}

		while(true) {
			uint32_t target = index;
			left_child = (index << 1) + 1;
			right_child = left_child + 1;
			if(right_child < els && data[right_child] < data[index]) {
				target = right_child;
			}
			if(left_child < els && data[left_child] < data[target]) {
				target = left_child;
			}
			if(target == index) {
				break;
			}
			hashmaps[i].erase(data[index].key, index, hashmaps_size);
			hashmaps[i].erase(data[target].key, target, hashmaps_size);
			swap<HeapEntry<T, U>>(data + index, data + target);
			hashmaps[i].add(data[index].key, index, hashmaps_size);
			hashmaps[i].add(data[target].key, target, hashmaps_size);
			index = target;
		}
		return result;
	}
public:
	__host__ HeapsDevice(const HeapsHost<T, U, minimum_free>& b) : HeapsHelper::HeapsBase<T, U>(b), hashmaps(b.hashmaps->get_device_array()), hashmaps_size(b.hashmaps->size) {}
	HeapsDevice() = delete;

	__device__ HeapEntry<T, U> extract(uint32_t i) {
		return remove(i, 0);
	}

	__device__ bool empty(uint32_t i) const {
		return !(elements[i]);
	}

	__device__ void insert(uint32_t i, const HeapEntry<T, U>& el) {
		HeapEntry<T, U>* data = HeapsHelper::get_heap(data_device, i, max_elements);
		uint32_t index = elements[i];
		data[index] = el;
		++elements[i];
		uint32_t parent;
		hashmaps[i].add(el.key, index, hashmaps_size);
		while(index) {
			parent = (index - 1) >> 1;
			if(data[index] < data[parent]) {
				hashmaps[i].erase(data[index].key, index, hashmaps_size);
				hashmaps[i].erase(data[parent].key, parent, hashmaps_size);
				swap<HeapEntry<T, U>>(data + index, data + parent);
				hashmaps[i].add(data[index].key, index, hashmaps_size);
				hashmaps[i].add(data[parent].key, parent, hashmaps_size);
				index = parent;
			} else {
				break;
			}
		}
	}

	__device__ void erase(uint32_t i, const T& key) {
		HeapEntry<T, U>* data = HeapsHelper::get_heap(data_device, i, max_elements);
		DeviceHashmapHelper::HashBucket<uint32_t, HASHMAP_BUCKET_DEPTH>* indices = hashmaps[i].get(key, hashmaps_size);
		uint32_t index = 0;
		for(; index < indices->total_used; ++index) {
			if(data[indices->values[index]].key == key) {
				break;
			}
		}

		if(index >= indices->total_used) {
			return;
		}

		remove(i, indices->values[index]);
	}
};

template<typename T, typename U, uint16_t minimum_free>
__global__ void verify_heap_hashmap(HeapsDevice<T, U, minimum_free> heaps) {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index >= heaps.get_heaps_count()) {
		return;
	}
}

template<typename T, typename U, uint16_t minimum_free>
class HeapsHost : public HeapsHelper::HeapsBase<T, U> {
	friend class HeapsDevice<T, U, minimum_free>;
	using HeapsHelper::HeapsBase<T, U>::DEFAULT_MAX_ELEMENTS;
	using HeapsHelper::HeapsBase<T, U>::HASHMAP_BUCKET_DEPTH;
	using HeapsHelper::HeapsBase<T, U>::heaps_count;
	using HeapsHelper::HeapsBase<T, U>::elements;
	using HeapsHelper::HeapsBase<T, U>::data_device;
	using HeapsHelper::HeapsBase<T, U>::max_elements;
	HeapEntry<T, U>** data_host;
	HeapEntry<T, U>** tmp_data_host;
	HeapEntry<T, U>** tmp_data_device;
	U* min_tmp;
	DeviceHashmapHost<T, uint32_t, HASHMAP_BUCKET_DEPTH, minimum_free>* hashmaps;

	__host__ uint32_t max_occupied_spaces() const {
		return reduction<uint32_t, max<uint32_t>>(elements, heaps_count);
	}

	__host__ uint32_t expand_all(uint32_t new_min_size) {
		if(new_min_size <= max_elements) {
			return max_elements;
		}
		uint32_t entries = heaps_count / MEMORY_CHUNK_SIZE;
		uint32_t new_max_elements = max_elements;
		while(new_min_size > new_max_elements) {
			new_max_elements <<= 1;
		}
		for(uint32_t i = 0; i < entries; ++i) {
			cuda_check(cudaMalloc(tmp_data_host + i, new_max_elements * sizeof(HeapEntry<T, U>) * MEMORY_CHUNK_SIZE));
		}
		cuda_check(cudaMemcpy(tmp_data_device, tmp_data_host, entries * sizeof(HeapEntry<T, U>*), cudaMemcpyHostToDevice));
		HeapsHelper::copy_data<<<get_block_count(max_elements * heaps_count), BLOCK_SIZE>>>(max_elements, heaps_count, data_device, tmp_data_device);
		for(uint32_t i = 0; i < entries; ++i) {
			cuda_check(cudaFree(data_host[i]));
		}
		max_elements = new_max_elements;
		std::swap(data_host, tmp_data_host);
		std::swap(data_device, tmp_data_device);
		return new_max_elements;
	}

	__host__ void populate_empty_hashmaps(){
		HeapsHelper::insert_into_hashmaps<<<get_block_count(heaps_count * max_elements), BLOCK_SIZE>>>(elements, data_device, heaps_count, max_elements, hashmaps->get_device_array(), hashmaps->size);
	}

public:
	__host__ __device__ HeapsHost() = delete;
	__host__ __device__ HeapsHost(const HeapsHost&) = delete;
	__host__ __device__ HeapsHost(HeapsHost&&) = delete;

	__host__ HeapsHost(uint32_t count) {
		if(count % MEMORY_CHUNK_SIZE) {
			throw std::runtime_error("Heap count has to be divisible by MEMORY_CHUNK_SIZE, which is " + std::to_string(MEMORY_CHUNK_SIZE));
		}
		hashmaps = new DeviceHashmapHost<T, uint32_t, HASHMAP_BUCKET_DEPTH, minimum_free>(count, DEFAULT_MAX_ELEMENTS);
		uint32_t entries = count / MEMORY_CHUNK_SIZE;
		heaps_count = count;
		max_elements = DEFAULT_MAX_ELEMENTS;
		data_host = new HeapEntry<T, U>*[entries];
		tmp_data_host = new HeapEntry<T, U>*[entries];

		cuda_check(cudaMalloc(&min_tmp, count * sizeof(U)));
		cuda_check(cudaMalloc(&elements, count * sizeof(uint32_t)));
		cuda_check(cudaMemset(elements, 0, count * sizeof(uint32_t)));
		for(uint32_t i = 0; i < entries; ++i) {
			cuda_check(cudaMalloc(data_host + i, max_elements * sizeof(HeapEntry<T, U>) * MEMORY_CHUNK_SIZE));
		}
		cuda_check(cudaMalloc(&(this->tmp_data_device), entries * sizeof(HeapEntry<T, U>*)));
		cuda_check(cudaMalloc(&(this->data_device), entries * sizeof(HeapEntry<T, U>*)));
		cuda_check(cudaMemcpy(this->data_device, data_host, entries * sizeof(HeapEntry<T, U>*), cudaMemcpyHostToDevice));
	}

	__host__ ~HeapsHost() {
		delete hashmaps;
		cuda_check(cudaFree(min_tmp));
		cuda_check(cudaFree(tmp_data_device));
		delete[] tmp_data_host;
		cuda_check(cudaFree(this->data_device));
		for(uint32_t i = 0; i < heaps_count / MEMORY_CHUNK_SIZE; ++i) {
			cuda_check(cudaFree(data_host[i]));
		}
		delete[] data_host;
		cuda_check(cudaFree(elements));
	}

	__host__ void assure_free() {
		verify_heap_hashmap<T, U, minimum_free><<<get_block_count(heaps_count), BLOCK_SIZE>>>(*this);
		expand_all(max_occupied_spaces() + minimum_free);
		verify_heap_hashmap<T, U, minimum_free><<<get_block_count(heaps_count), BLOCK_SIZE>>>(*this);
		while(hashmaps->full()) {
			uint32_t old_hashmaps_size = hashmaps->size;
			delete hashmaps;
			hashmaps = new DeviceHashmapHost<T, uint32_t, HASHMAP_BUCKET_DEPTH, minimum_free>(heaps_count, old_hashmaps_size << 1);
			populate_empty_hashmaps();
		}
		verify_heap_hashmap<T, U, minimum_free><<<get_block_count(heaps_count), BLOCK_SIZE>>>(*this);
	}

	__host__ uint32_t size() const {
		return reduction<uint32_t, sum<uint32_t>>(elements, heaps_count);
	}

	__host__ U minimum(const U& default_element) {
		HeapsHelper::get_min_elements<T><<<get_block_count(heaps_count), BLOCK_SIZE>>>(min_tmp, elements, data_device, default_element, heaps_count, max_elements);
		return reduction<U, min<U>>(min_tmp, heaps_count);
	}
};

#endif
