#include "a_star_sliding_puzzle.h"
#include "a_star_pathfinding.h"

#include <iostream>
#include <fstream>
#include <string>
#include <functional>

template<typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& vec){
	for(const T& el : vec) {
		o<<el<<'\n';
	}
	return o;
}

const int CUDA_DEVICE = 1;
const int32_t NUM_QUEUES = 1 << 12;

template<typename T>
void error(const T& msg) {
	std::cout<<msg<<std::endl;
	exit(-1);
}

template<typename T>
void print_result(std::ostream& out, const T& result) {
	out << std::chrono::duration_cast<std::chrono::milliseconds>(result.second).count() << '\n' << result.first;
}

int main(int argv, char* argc[]) {
	std::string input;
	std::string output;
	std::string version;
	for(int i = 1; i < argv; i += 2) {
		if(i == argv - 1) {
			error("Parameter without argument.");
		}
		if(!strcmp(argc[i], "--version")) {
			version = argc[i + 1];
			continue;
		} else if(!strcmp(argc[i], "--input-data")) {
			input = argc[i + 1];
			continue;
		} else if(!strcmp(argc[i], "--output-data")) {
			output = argc[i + 1];
			continue;
		} else {
			error(std::string("Unknown option ") + argc[i]);
		}
	}

	std::ifstream in_stream;
	std::ofstream out_stream;

	std::istream& in = input.empty() ? std::cin : in_stream;
	std::ostream& out = output.empty() ? std::cout : out_stream;

	in.exceptions(std::ios_base::failbit | std::ios_base::badbit);
	out.exceptions(std::ios_base::failbit | std::ios_base::badbit);

	if(!input.empty()) {
		in_stream.open(input);
	}
	if(!output.empty()) {
		out_stream.open(output);
	}

	cudaSetDevice(CUDA_DEVICE);

	if(version == "sliding") {
		print_result(out, SlidingPuzzle::solve<5>(in, NUM_QUEUES));
	} else if(version == "pathfinding") {
		print_result(out, Pathfinding::solve(in, NUM_QUEUES));
	} else {
		error("Unknown version " + version);
	}
}
