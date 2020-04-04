astar_gpu : a_star.h a_star_sliding_puzzle.h bitonic_sort.h consts.h device_hashmap.h fancy_hashmap.h functional_types.h heaps.h helpers.h main.cu reduction.h a_star_pathfinding.h
	nvcc main.cu -o astar_gpu -std=c++14 -gencode arch=compute_61,code=sm_61 -O3
