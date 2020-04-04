#ifndef A_STAR_H
#define A_STAR_H

#include "heaps.h"
#include "reduction.h"
#include "fancy_hashmap.h"
#include "functional_types.h"
#include "helpers.h"
#include "consts.h"
#include "bitonic_sort.h"

#include <vector>
#include <algorithm>
#include <chrono>
#include <utility>

namespace AStarHelper {
	const uint32_t DEFAULT_HASHMAP_SIZE = 8096;

	template<typename T>
	struct FancyHashmapValue {
		T prev;
		uint32_t distance_from_start;
		uint32_t queue_number;
		__host__ __device__ bool operator<(const FancyHashmapValue& o) const {
			return distance_from_start < o.distance_from_start;
		}
		__host__ __device__ bool operator==(const FancyHashmapValue& o) const {
			return distance_from_start == o.distance_from_start;
		}
		__host__ __device__ bool operator<=(const FancyHashmapValue& o) const {
			return distance_from_start <= o.distance_from_start;
		}
		//NIE USTAWIAMY distance_from_start BO PO WŁOŻENIU DO HASHMAPY STARA WARTOŚĆ NIE JEST POTRZEBNA, ZAŚ W insert_nodes_into_queues KORZYSTAMY Z NOWEJ
		__device__ void set_old(const FancyHashmapValue& o){
			prev = o.prev;
			queue_number = o.queue_number;
		}
	};

	template<typename T>
	struct FancyHashmapValueForByQueueSort : private FancyHashmapValue<T> {
		using FancyHashmapValue<T>::queue_number;
		__host__ __device__ bool operator<(const FancyHashmapValueForByQueueSort<T>& o) const {
			return queue_number < o.queue_number;
		}
		__host__ __device__ FancyHashmapValueForByQueueSort() = delete;
	};

	template<typename T>
	FancyHashmapValueForByQueueSort<T>* cast_hash_values_for_by_queue_sort(FancyHashmapValue<T>* f) {
		return (FancyHashmapValueForByQueueSort<T>*) f;
	}

	struct QueueValue {
		uint32_t total_cost;
		uint32_t distance_from_start;
		__device__ bool operator<(const QueueValue& o) const {
			if(total_cost == o.total_cost) {
				return distance_from_start > o.distance_from_start;
			}
			return total_cost < o.total_cost;
		}
		__host__ __device__ QueueValue() = delete;
		__device__ QueueValue(uint32_t total_cost, uint32_t distance_from_start) : total_cost(total_cost), distance_from_start(distance_from_start) {}
	};

	template<typename T, Heuristic<T> heuristic, unsigned char max_neighbours>
	__global__ void insert_first(HeapsDevice<T, QueueValue, max_neighbours> queues, T value){
		queues.insert(0, HeapEntry<T, QueueValue>(value, QueueValue(heuristic(value), 0)));
	}

	template<typename T, unsigned char max_neighbours, NeighbourGenerator<T> neighbour_generator, Heuristic<T> heuristic>
	__global__ void extract_nodes(HeapsDevice<T, QueueValue, max_neighbours> queues, T* s_element, unsigned char* s_status, uint32_t* m_distance, const T target, AStarHelper::FancyHashmapValue<T>* hash_values, uint32_t* s_total_length){
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index >= queues.get_heaps_count() || queues.empty(index)) {
			return;
		}
		HeapEntry<T, QueueValue> element = queues.extract(index);
		if(element.key == target) {
			m_distance[index] = element.value.distance_from_start;
			return;
		}
		index = index * max_neighbours;
		for(int32_t i = 0; i < max_neighbours; ++i) {
			uint32_t ie = index + i;
			T& el = s_element[ie];
			bool ex;
			uint32_t cost;
			neighbour_generator(element.key, el, ex, cost, i);
			if(ex) {
				s_status[ie] = FANCY_HASHMAP_ENTRY_STATUS_NEW;
				uint32_t dist = element.value.distance_from_start + cost;
				s_total_length[ie] = heuristic(el) + dist;
				hash_values[ie].distance_from_start = dist;
				hash_values[ie].prev = element.key;
			}
		}
	}

	template<typename T, unsigned char max_neighbours, uint16_t bucket_depth>
	__global__ void insert_nodes_into_queues(HeapsDevice<T, QueueValue, max_neighbours> queues, const FancyHashmapValue<T>* hash_values, const T* s_element, unsigned char* s_status, uint32_t epoch, uint32_t* s_total_length) {
		uint32_t queue_index = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t hp = queues.get_heaps_count();
		if(queue_index >= hp) {
			return;
		}
		uint32_t index = (queue_index + epoch) % hp;
		uint32_t index_end = index + max_neighbours * hp;
		for(; index < index_end; index += hp) {
			if(s_status[index] == FANCY_HASHMAP_ENTRY_STATUS_INSERTED || s_status[index] == FANCY_HASHMAP_ENTRY_STATUS_UPDATED) {
				queues.insert(queue_index, HeapEntry<T, QueueValue>(s_element[index], QueueValue(s_total_length[index], hash_values[index].distance_from_start)));
			}
		}
	}

	template<typename T, unsigned char max_neighbours, uint16_t bucket_depth>
	__global__ void insert_nodes_into_hashmap(uint32_t heaps_count, const T* s_element, unsigned char* s_status, uint32_t epoch, uint32_t* s_total_length, FancyHashmapDevice<T, AStarHelper::FancyHashmapValue<T>, bucket_depth> hashmap) {
		uint32_t queue_index = blockIdx.x * blockDim.x + threadIdx.x;
		if(queue_index >= heaps_count) {
			return;
		}
		uint32_t index = (queue_index + epoch) % heaps_count;
		uint32_t index_end = index + max_neighbours * heaps_count;
		for(; index < index_end; index += heaps_count) {
			if(s_status[index] == FANCY_HASHMAP_ENTRY_STATUS_INSERTED || s_status[index] == FANCY_HASHMAP_ENTRY_STATUS_UPDATED) {
				hashmap.get(s_element[index])->queue_number = queue_index;
			}
		}
	}

	template<typename T, uint16_t bucket_depth>
	__global__ void get_prev(T el, T* destination, FancyHashmapDevice<T, FancyHashmapValue<T>, bucket_depth> hashmap){
		*destination = hashmap.get(el)->prev;
	}

	template<typename T>
	__global__ void mark_duplicates(uint32_t element_count, const T* s_element, unsigned char* s_status){
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index == 0 || index >= element_count || s_status[index] != FANCY_HASHMAP_ENTRY_STATUS_NEW) {
			return;
		}
		if(s_element[index] == s_element[index - 1]) {
			s_status[index] = FANCY_HASHMAP_ENTRY_STATUS_EMPTY;
		}
	}

	template<typename T>
	__device__ uint32_t lower_bound_queue_num(const AStarHelper::FancyHashmapValue<T>* hash_values, const uint32_t element_count, uint32_t queue_num){
		uint32_t low = 0;
		uint32_t high = element_count;
		while(low < high) {
			uint32_t mid = low + ((high - low) >> 1);
			if(queue_num <= hash_values[mid].queue_number) {
				high = mid;
			} else {
				low = mid + 1;
			}
		}
		return low;
	}

	template<typename T>
	__device__ uint32_t upper_bound_queue_num(const AStarHelper::FancyHashmapValue<T>* hash_values, const uint32_t element_count, uint32_t queue_num){
		uint32_t low = 0;
		uint32_t high = element_count;
		while(low < high) {
			uint32_t mid = low + ((high - low) >> 1);
			if(queue_num < hash_values[mid].queue_number) {
				high = mid;
			} else {
				low = mid + 1;
			}
		}
		return low;
	}

	__device__ uint32_t upper_bound_updated_reversed(const unsigned char* s_status, const uint32_t element_count) {
		uint32_t low = 0;
		uint32_t high = element_count;
		while(low < high) {
			uint32_t mid = (low + high) >> 1;
			if(FANCY_HASHMAP_ENTRY_STATUS_UPDATED > s_status[mid]) {
				high = mid;
			} else {
				low = mid + 1;
			}
		}
		return low;
	}

	template<typename T, unsigned char max_neighbours>
	__global__ void remove_updated_elements(HeapsDevice<T, QueueValue, max_neighbours> queues, const uint32_t element_count, const T* s_element, const unsigned char* s_status, const FancyHashmapValue<T>* hash_values){
		uint32_t updated_size = upper_bound_updated_reversed(s_status, element_count);
		uint32_t queue_num = blockIdx.x * blockDim.x + threadIdx.x;
		if(queue_num >= queues.get_heaps_count()) {
			return;
		}
		uint32_t limit = upper_bound_queue_num(hash_values, updated_size, queue_num);
		for(uint32_t el_index = lower_bound_queue_num(hash_values, updated_size, queue_num); el_index < limit; ++el_index) {
			queues.erase(queue_num, s_element[el_index]);
		}
	}

	template<typename T>
	__host__ T* prev_get_tmp() {
		static T* tmp = nullptr;
		if(!tmp) {
			cuda_check(cudaMalloc(&tmp, sizeof(T)));
		}
		return tmp;
	}

	template<typename T, uint16_t bucket_depth>
	__host__ T get_prev(const T& el, FancyHashmapDevice<T, AStarHelper::FancyHashmapValue<T>, bucket_depth> hashmap){
		unsigned char value[sizeof(T)];
		get_prev<T><<<1, 1>>>(el, prev_get_tmp<T>(), hashmap);
		cuda_check(cudaMemcpy(value, prev_get_tmp<T>(), sizeof(T), cudaMemcpyDeviceToHost));
		return *((T*) value);
	}
}

template<typename T, unsigned char max_neighbours, NeighbourGenerator<T> neighbour_generator, Heuristic<T> heuristic, uint16_t bucket_depth = 256>
std::pair<std::vector<T>, std::chrono::nanoseconds> a_star(const T& start, const T& target, const uint32_t num_queues){
	HeapsHost<T, AStarHelper::QueueValue, max_neighbours> queues(num_queues);
	AStarHelper::insert_first<T, heuristic, max_neighbours><<<1, 1>>>(queues, start);
	uint32_t total_padded_size = get_nearest_power_of_two(num_queues * max_neighbours);
	uint32_t* m_distance;
	cuda_check(cudaMalloc(&m_distance, sizeof(uint32_t) * num_queues));
	cuda_check(cudaMemset(m_distance, 0xff, sizeof(uint32_t) * num_queues));
	T* s_element;
	cuda_check(cudaMalloc(&s_element, sizeof(T) * total_padded_size));
	unsigned char* s_status;
	cuda_check(cudaMalloc(&s_status, sizeof(bool) * total_padded_size));
	uint32_t* s_total_length;
	cuda_check(cudaMalloc(&s_total_length, sizeof(uint32_t) * total_padded_size));
	AStarHelper::FancyHashmapValue<T>* hash_values;
	cuda_check(cudaMalloc(&hash_values, sizeof(AStarHelper::FancyHashmapValue<T>) * total_padded_size));

	AStarHelper::QueueValue default_value(-1U, -1U);

	FancyHashmapHost<T, AStarHelper::FancyHashmapValue<T>, bucket_depth> hashed(AStarHelper::DEFAULT_HASHMAP_SIZE);

	uint32_t epoch = 0;
	auto begin_time = std::chrono::steady_clock::now();

	while(queues.size() != 0) {
		cuda_check(cudaMemset(s_status, FANCY_HASHMAP_ENTRY_STATUS_EMPTY, sizeof(bool) * total_padded_size));
		AStarHelper::extract_nodes<T, max_neighbours, neighbour_generator, heuristic><<<get_block_count(num_queues), BLOCK_SIZE>>>(queues, s_element, s_status, m_distance, target, hash_values, s_total_length);
		uint32_t m_dist = reduction<uint32_t, min<uint32_t>>(m_distance, num_queues);
		bitonic_sort(total_padded_size, BitonicReverse<unsigned char>(s_status), s_element, hash_values, BitonicDontSort(), s_total_length);
		AStarHelper::mark_duplicates<<<get_block_count(num_queues * max_neighbours), BLOCK_SIZE>>>(num_queues * max_neighbours, s_element, s_status);
		hashed.insert_unique_data(s_element, hash_values, s_status, num_queues * max_neighbours);
		bitonic_sort(total_padded_size, BitonicReverse<unsigned char>(s_status), AStarHelper::cast_hash_values_for_by_queue_sort(hash_values), BitonicDontSort(), s_total_length, s_element);
		AStarHelper::remove_updated_elements<T, max_neighbours><<<get_block_count(num_queues), BLOCK_SIZE>>>(queues, num_queues * max_neighbours, s_element, s_status, hash_values);
		queues.assure_free();
		bitonic_sort(total_padded_size, BitonicReverse<unsigned char>(s_status), s_total_length, BitonicDontSort(), s_element, hash_values);
		AStarHelper::insert_nodes_into_queues<T, max_neighbours, bucket_depth><<<get_block_count(num_queues), BLOCK_SIZE>>>(queues, hash_values, s_element, s_status, epoch, s_total_length);
		AStarHelper::insert_nodes_into_hashmap<T, max_neighbours, bucket_depth><<<get_block_count(num_queues), BLOCK_SIZE>>>(queues.get_heaps_count(), s_element, s_status, epoch, s_total_length, hashed);
		if(m_dist != -1U && m_dist <= queues.minimum(default_value).total_cost) {
			break;
		}
		++epoch;
	}

	auto end_time = std::chrono::steady_clock::now();

	std::vector<T> result;
	if(reduction<uint32_t, min<uint32_t>>(m_distance, num_queues) != -1U) {
		result.push_back(target);
		while(!(result[result.size() - 1] == start)) {
			result.push_back(AStarHelper::get_prev<T, bucket_depth>(result[result.size() - 1], hashed));
		}
	}
	cuda_check(cudaFree(hash_values));
	cuda_check(cudaFree(s_total_length));
	cuda_check(cudaFree(s_status));
	cuda_check(cudaFree(s_element));
	cuda_check(cudaFree(m_distance));
	std::reverse(result.begin(), result.end());
	return std::make_pair(std::move(result), end_time - begin_time);
}

#endif
