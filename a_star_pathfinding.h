#ifndef A_STAR_PATHFINDING_H
#define A_STAR_PATHFINDING_H

#include "a_star.h"
#include "helpers.h"

#include <istream>
#include <ostream>
#include <string>
#include <vector>
#include <tuple>

namespace Pathfinding {
	struct Position;
	__host__ Position scan_coordinates(std::istream& in);

	struct Position {
		uint16_t x;
		uint16_t y;

		explicit Position() = default;
		__host__ Position(std::istream& in) : Position(scan_coordinates(in)) {}
		__host__ Position(uint16_t x, uint16_t y) : x(x), y(y) {}

		__host__ __device__ bool operator==(const Position& o) const {
			return (*((uint32_t*) this)) == (*((uint32_t*) &o));
		}

		__host__ __device__ bool operator<(const Position& o) const {
			return (*((uint32_t*) this)) < (*((uint32_t*) &o));
		}

		__host__ __device__ bool operator<=(const Position& o) const {
			return (*((uint32_t*) this)) <= (*((uint32_t*) &o));
		}
	};

	__host__ Position scan_coordinates(std::istream& in){
		std::string str;
		in >> str;
		auto delim = str.find(',');
		if(delim >= str.size()) {
			throw std::runtime_error("Pahtfinding incorrect input.");
		}
		uint16_t x = std::stoi(str.substr(0, delim));
		uint16_t y = std::stoi(str.substr(delim + 1));
		return Position(x, y);
	}

	__host__ std::tuple<uint16_t, uint16_t, uint16_t> scan_triple_int(std::istream& in){
		std::string str;
		in >> str;
		auto delim = str.find(',');
		if(delim >= str.size()) {
			throw std::runtime_error("Pahtfinding incorrect input.");
		}
		auto delim2 = str.find(',', delim + 1);
		if(delim2 >= str.size()) {
			throw std::runtime_error("Pahtfinding incorrect input.");
		}
		uint16_t x = std::stoi(str.substr(0, delim));
		uint16_t y = std::stoi(str.substr(delim + 1, delim2 - delim));
		uint16_t z = std::stoi(str.substr(delim2 + 1));
		return std::make_tuple(x, y, z);
	}

	__host__ std::ostream& operator<<(std::ostream& o, const Position& p) {
		o << p.x << ',' << p.y;
		return o;
	}

	__constant__ Position manhattan_heuristic_data;

	__constant__ Position* neighbour_generator_costs_keys;
	__constant__ unsigned char* neighbour_generator_costs_values;
	__constant__ uint32_t neighbour_generator_costs_size;

	__constant__ Position* neighbour_generator_obstacles;
	__constant__ uint32_t neighbour_generator_obstacles_size;

	__constant__ Position neighbour_generator_board_size;

	__host__ void fill_generator_data(std::istream& in, Position*& gpu_obstacles, Position*& gpu_costs_keys, unsigned char*& gpu_costs_values) {
		int32_t obstacle_count;
		in >> obstacle_count;
		cuda_check(cudaMalloc(&gpu_obstacles, obstacle_count * sizeof(Position)));
		cuda_check(cudaMemcpyToSymbol(neighbour_generator_obstacles_size, &obstacle_count, sizeof(obstacle_count)));
		cuda_check(cudaMemcpyToSymbol(neighbour_generator_obstacles, &gpu_obstacles, sizeof(gpu_obstacles)));

		std::vector<Position> obstacles;
		obstacles.reserve(obstacle_count);
		for(int32_t i = 0; i < obstacle_count; ++i) {
			obstacles.emplace_back(in);
		}
		std::sort(obstacles.begin(), obstacles.end());
		cuda_check(cudaMemcpy(gpu_obstacles, obstacles.data(), obstacle_count * sizeof(Position), cudaMemcpyHostToDevice));

		int32_t cost_count;
		in >> cost_count;
		cuda_check(cudaMalloc(&gpu_costs_keys, cost_count * sizeof(Position)));
		cuda_check(cudaMalloc(&gpu_costs_values, cost_count * sizeof(unsigned char)));
		cuda_check(cudaMemcpyToSymbol(neighbour_generator_costs_keys, &gpu_costs_keys, sizeof(gpu_costs_keys)));
		cuda_check(cudaMemcpyToSymbol(neighbour_generator_costs_values, &gpu_costs_values, sizeof(gpu_costs_values)));
		cuda_check(cudaMemcpyToSymbol(neighbour_generator_costs_size, &cost_count, sizeof(cost_count)));

		std::vector<std::tuple<uint16_t, uint16_t, uint16_t>> costs;
		costs.reserve(cost_count);
		std::vector<Position> cost_keys;
		cost_keys.reserve(cost_count);
		std::vector<unsigned char> cost_values;
		cost_values.reserve(cost_count);

		for(int32_t i = 0; i < cost_count; ++i) {
			costs.push_back(scan_triple_int(in));
		}

		std::sort(costs.begin(), costs.end(), [](const std::tuple<uint16_t, uint16_t, uint16_t>& a, const std::tuple<uint16_t, uint16_t, uint16_t>& b){
			return Position(std::get<0>(a), std::get<1>(a)) < Position(std::get<0>(b), std::get<1>(b));
		});

		for(int32_t i = 0; i < cost_count; ++i) {
			cost_keys.push_back(Position(std::get<0>(costs[i]), std::get<1>(costs[i])));
			cost_values.push_back(std::get<2>(costs[i]));
		}

		cuda_check(cudaMemcpy(gpu_costs_keys, cost_keys.data(), cost_count * sizeof(Position), cudaMemcpyHostToDevice));
		cuda_check(cudaMemcpy(gpu_costs_values, cost_values.data(), cost_count * sizeof(unsigned char), cudaMemcpyHostToDevice));
	}

	__device__ uint32_t find_position(const Position& target, uint32_t size, const Position* data){
		uint32_t low = 0;
		uint32_t high = size;
		while(low < high) {
			uint32_t mid = low + ((high - low) >> 1);
			if(target <= data[mid]) {
				high = mid;
			} else {
				low = mid + 1;
			}
		}
		if(low < size && target == data[low]) {
			return low;
		}
		return -1U;
	}

	__device__ void neighbour_generator(const Position& source, Position& target, bool& target_exists, uint32_t& cost, unsigned char i) {
		uint16_t delta_x;
		uint16_t delta_y;
		switch(i) {
		case 0:
			delta_x = 1;
			delta_y = 0;
			break;
		case 1:
			delta_x = 0xffff;
			delta_y = 0;
			break;
		case 2:
			delta_x = 0;
			delta_y = 1;
			break;
		case 3:
			delta_x = 0;
			delta_y = 0xffff;
			break;
		case 4:
			delta_x = 1;
			delta_y = 1;
			break;
		case 5:
			delta_x = 1;
			delta_y = 0xffff;
			break;
		case 6:
			delta_x = 0xffff;
			delta_y = 1;
			break;
		case 7:
			delta_x = 0xffff;
			delta_y = 0xffff;
			break;
		}

		unsigned char nposx = source.x + delta_x;
		unsigned char nposy = source.y + delta_y;

		if(nposx >= neighbour_generator_board_size.x || nposy >= neighbour_generator_board_size.y) {
			target_exists = false;
			return;
		}

		target.x = nposx;
		target.y = nposy;

		if(find_position(target, neighbour_generator_obstacles_size, neighbour_generator_obstacles) != -1U) {
			target_exists = false;
			return;
		}

		target_exists = true;

		uint32_t cost_p = find_position(target, neighbour_generator_costs_size, neighbour_generator_costs_keys);
		if(cost_p != -1U) {
			cost = neighbour_generator_costs_values[cost_p];
		} else {
			cost = 1;
		}
	}

	template<typename T>
	__device__ T abs(const T& a) {
		if(a < 0) {
			return -a;
		}
		return a;
	}

	__device__ uint32_t manhattan_heuristic(const Position& p) {
		return abs(manhattan_heuristic_data.x - p.x) + abs(manhattan_heuristic_data.y - p.y);
	}

	__host__ std::pair<std::vector<Position>, std::chrono::nanoseconds> solve(std::istream& in, uint32_t num_queues) {
		Position board_size(in);
		cuda_check(cudaMemcpyToSymbol(neighbour_generator_board_size, &board_size, sizeof(board_size)));
		Position start(in);
		Position target(in);
		Position* gpu_obstacles;
		Position* gpu_costs_keys;
		unsigned char* gpu_costs_values;
		fill_generator_data(in, gpu_obstacles, gpu_costs_keys, gpu_costs_values);
		cuda_check(cudaMemcpyToSymbol(manhattan_heuristic_data, &target, sizeof(target)));
		auto r = a_star<Position, 8, neighbour_generator, manhattan_heuristic>(start, target, num_queues);
		cuda_check(cudaFree(gpu_obstacles));
		cuda_check(cudaFree(gpu_costs_keys));
		cuda_check(cudaFree(gpu_costs_values));
		return r;
	}
}

#endif
