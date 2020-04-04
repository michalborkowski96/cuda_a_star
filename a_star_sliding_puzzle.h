#ifndef A_STAR_SLIDING_PUZZLE_H
#define A_STAR_SLIDING_PUZZLE_H

#include "a_star.h"
#include "helpers.h"

#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace SlidingPuzzle {
	template<unsigned char size>
	struct Position {
		unsigned char pos_zero_x;
		unsigned char pos_zero_y;
		unsigned char board[size][size];
		unsigned char padding[(4 - ((size * size + 2) % 4)) % 4];

		__host__ __device__ Position() = delete;
		__host__ Position(std::istream& in) {
			std::string str;
			in >> str;
			size_t bpos = 0;
			size_t pos = 0;
			std::string num;
			std::vector<int> nums;
			for(int i = 0; i < size * size; ++i) {
				pos = str.find(',', bpos);
				num = str.substr(bpos, pos - bpos);
				if(num == "_") {
					num = '0';
				}
				nums.push_back(std::stoi(num));
				bpos = pos + 1;
			}
			for(unsigned char i = 0; i < size; ++i) {
				for(unsigned char j = 0; j < size; ++j) {
					int a = nums[i * size + j];
					board[i][j] = a;
					if(a == 0) {
						pos_zero_x = i;
						pos_zero_y = j;
					}
				}
			}
			fill_padding(*this);
		}

		__host__ __device__ bool operator==(const Position& o) const {
			return mem_eq<Position<size>>(this, &o);
		}

		__host__ __device__ bool operator<(const Position& o) const {
			return mem_less<Position<size>>(this, &o);
		}
	};

	template<unsigned char size>
	__host__ std::ostream& operator<<(std::ostream& o, const Position<size>& p) {
		for(unsigned char i = 0; i < size; ++i) {
			for(unsigned char j = 0; j < size; ++j) {
				if(p.board[i][j]) {
					o << (int) p.board[i][j];
				} else {
					o << '_';
				}
				if((i != (size - 1)) || (j != (size - 1))) {
					o << ',';
				}
			}
		}
		return o;
	}

	template<unsigned char size>
	__host__ __device__ void fill_padding(Position<size>& p) {
		for(unsigned char* a = p.padding; a < (unsigned char*) ((&p) + 1); ++a) {
			*a = 69;
		}
	}

	template<unsigned char size>
	__device__ void neighbour_generator(const Position<size>& source, Position<size>& target, bool& target_exists, uint32_t& cost, unsigned char i) {
		cost = 1;
		unsigned char delta_x;
		unsigned char delta_y;
		switch(i) {
		case 0:
			delta_x = 1;
			delta_y = 0;
			break;
		case 1:
			delta_x = 0xff;
			delta_y = 0;
			break;
		case 2:
			delta_x = 0;
			delta_y = 1;
			break;
		case 3:
			delta_x = 0;
			delta_y = 0xff;
			break;
		}
		uint16_t nposx = source.pos_zero_x + delta_x;
		uint16_t nposy = source.pos_zero_y + delta_y;
		if(nposx >= size || nposy >= size) {
			target_exists = false;
			return;
		}
		target_exists = true;
		target.pos_zero_x = nposx;
		target.pos_zero_y = nposy;
		for(int i = 0; i < size; ++i) {
			for(int j = 0; j < size; ++j) {
				target.board[i][j] = source.board[i][j];
			}
		}
		target.board[source.pos_zero_x][source.pos_zero_y] = source.board[nposx][nposy];
		target.board[nposx][nposy] = 0;
		fill_padding(target);
	}

	template<unsigned char size>
	__constant__ signed char manhattan_heuristic_data_x[size * size];
	template<unsigned char size>
	__constant__ signed char manhattan_heuristic_data_y[size * size];

	template<unsigned char size>
	__host__ void set_manhattan_heuristic_target(const Position<size>& target) {
		unsigned char data_x[size * size];
		unsigned char data_y[size * size];
		for(unsigned char i = 0; i < size; ++i) {
			for(unsigned char j = 0; j < size; ++j) {
				data_x[target.board[i][j]] = i;
				data_y[target.board[i][j]] = j;
			}
		}
		cuda_check(cudaMemcpyToSymbol(manhattan_heuristic_data_x<size>, data_x, sizeof(data_x)));
		cuda_check(cudaMemcpyToSymbol(manhattan_heuristic_data_y<size>, data_y, sizeof(data_y)));
	}

	template<typename T>
	__device__ T abs(const T& a) {
		if(a < 0) {
			return -a;
		}
		return a;
	}

	template<unsigned char size>
	__device__ uint32_t manhattan_heuristic(const Position<size>& p) {
		uint32_t score = 0;
		for(int i = 0; i < size; ++i) {
			for(int j = 0; j < size; ++j) {
				unsigned char a = p.board[i][j];
				score += abs(manhattan_heuristic_data_x<size>[a] - i);
				score += abs(manhattan_heuristic_data_y<size>[a] - j);
			}
		}
		return score;
	}

	template<unsigned char size>
	__host__ std::pair<std::vector<Position<size>>, std::chrono::nanoseconds> solve(std::istream& in, uint32_t num_queues) {
		Position<size> start(in);
		Position<size> target(in);
		set_manhattan_heuristic_target(target);
		return a_star<Position<size>, 4, neighbour_generator<size>, manhattan_heuristic<size>>(start, target, num_queues);
	}
}

#endif
