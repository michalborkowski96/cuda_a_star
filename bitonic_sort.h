#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H

#include "helpers.h"

//https://en.wikipedia.org/wiki/Bitonic_sorter

struct BitonicDontSort {};

template<typename T>
struct BitonicReverse {
	T* data;
	__host__ __device__ BitonicReverse() = delete;
	__host__ __device__ BitonicReverse(T* data) : data(data) {}
};

template <typename T, std::enable_if_t<is_instance<T, BitonicReverse>::value, int>*>
__device__ bool bitonic_sort_less(uint32_t i1, uint32_t i2, T data);

template <typename T, std::enable_if_t<std::is_pointer<T>::value, int>*>
__device__ bool bitonic_sort_less(uint32_t i1, uint32_t i2, T data);

template <typename T, std::enable_if_t<is_same_kind<T, BitonicDontSort>::value, int>*>
__device__ bool bitonic_sort_less(uint32_t i1, uint32_t i2, T data);

template <typename T, std::enable_if_t<is_same_kind<T, BitonicDontSort>::value, int>*, typename... Ts>
__device__ bool bitonic_sort_less(uint32_t i1, uint32_t i2, T data, Ts... more_data);

template <typename T, std::enable_if_t<is_instance<T, BitonicReverse>::value, int>*, typename... Ts>
__device__ bool bitonic_sort_less(uint32_t i1, uint32_t i2, T data, Ts... more_data);

template <typename T, std::enable_if_t<std::is_pointer<T>::value, int>*, typename... Ts>
__device__ bool bitonic_sort_less(uint32_t i1, uint32_t i2, T data, Ts... more_data);






template <typename T, std::enable_if_t<is_instance<T, BitonicReverse>::value, int>* = nullptr>
__device__ bool bitonic_sort_less(uint32_t i1, uint32_t i2, T data) {
	return data.data[i2] < data.data[i1];
}

template <typename T, std::enable_if_t<std::is_pointer<T>::value, int>* = nullptr>
__device__ bool bitonic_sort_less(uint32_t i1, uint32_t i2, T data) {
	return data[i1] < data[i2];
}

template <typename T, std::enable_if_t<is_same_kind<T, BitonicDontSort>::value, int>* = nullptr>
__device__ bool bitonic_sort_less(uint32_t, uint32_t, T) {
	return false;
}

template <typename T, std::enable_if_t<is_same_kind<T, BitonicDontSort>::value, int>* = nullptr, typename... Ts>
__device__ bool bitonic_sort_less(uint32_t, uint32_t, T, Ts...) {
	return false;
}

template <typename T, std::enable_if_t<is_instance<T, BitonicReverse>::value, int>* = nullptr, typename... Ts>
__device__ bool bitonic_sort_less(uint32_t i1, uint32_t i2, T data, Ts... more_data) {
	if(data.data[i2] < data.data[i1]) {
		return true;
	}
	if(data.data[i1] < data.data[i2]) {
		return false;
	}
	return bitonic_sort_less(i1, i2, more_data...);
}

template <typename T, std::enable_if_t<std::is_pointer<T>::value, int>* = nullptr, typename... Ts>
__device__ bool bitonic_sort_less(uint32_t i1, uint32_t i2, T data, Ts... more_data) {
	if(data[i1] < data[i2]) {
		return true;
	}
	if(data[i2] < data[i1]) {
		return false;
	}
	return bitonic_sort_less(i1, i2, more_data...);
}

template<typename T, std::enable_if_t<is_same_kind<T, BitonicDontSort>::value, int>* = nullptr>
__device__ void bitonic_sort_swap_data_do(uint32_t, uint32_t, T){
}

template<typename T, std::enable_if_t<std::is_pointer<T>::value, int>* = nullptr>
__device__ void bitonic_sort_swap_data_do(uint32_t i1, uint32_t i2, T data){
	swap(data + i1, data + i2);
}

template<typename T, std::enable_if_t<is_instance<T, BitonicReverse>::value, int>* = nullptr>
__device__ void bitonic_sort_swap_data_do(uint32_t i1, uint32_t i2, T data){
	swap(data.data + i1, data.data + i2);
}

template<typename T>
__device__ void bitonic_sort_swap_data(uint32_t i1, uint32_t i2, T data){
	bitonic_sort_swap_data_do(i1, i2, data);
}

template<typename T, typename... Ts>
__device__ void bitonic_sort_swap_data(uint32_t i1, uint32_t i2, T data, Ts... next){
	bitonic_sort_swap_data_do(i1, i2, data);
	bitonic_sort_swap_data(i1, i2, next...);
}

template<typename... Ts>
__global__ void bitonic_sort_one_red_boxes_round_gpu(unsigned char red_box_size, unsigned char blue_box_size, Ts... data){
	uint32_t index = (blockIdx.x * blockDim.x + threadIdx.x) << 1;

	uint32_t red_box_begin = (index >> red_box_size) << red_box_size;
	uint32_t item_index = red_box_begin + ((index - red_box_begin) >> 1);
	uint32_t bigger_item_index = item_index + (1 << (red_box_size - 1));

	if((index >> blue_box_size) & 1) {
		uint32_t c = item_index;
		item_index = bigger_item_index;
		bigger_item_index = c;
	}

	if(bitonic_sort_less(bigger_item_index, item_index, data...)) {
		bitonic_sort_swap_data(item_index, bigger_item_index, data...);
	}
}
//red_box_size and blue_box_size are logarithm base two

template<typename... Ts>
void bitonic_sort_one_red_boxes_round(uint32_t data_size, unsigned char red_box_size, unsigned char blue_box_size, Ts... data){
	if(data_size > 64) {
		bitonic_sort_one_red_boxes_round_gpu<<<(data_size >> 7), 64>>>(red_box_size, blue_box_size, data...);
	} else {
		bitonic_sort_one_red_boxes_round_gpu<<<1, (data_size >> 1)>>>(red_box_size, blue_box_size, data...);
	}
}

template<typename... Ts>
void bitonic_sort(uint32_t data_size, Ts... data){
	for(unsigned char blue_box_size = 1; (1 << blue_box_size) <= data_size; ++blue_box_size) {
		for(unsigned char red_box_size = blue_box_size; red_box_size; --red_box_size) {
			bitonic_sort_one_red_boxes_round(data_size, red_box_size, blue_box_size, data...);
		}
	}
}

#endif
