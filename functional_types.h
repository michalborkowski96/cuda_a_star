#ifndef FUNCTIONAL_TYPES_H
#define FUNCTIONAL_TYPES_H

#include <cstdint>

template<typename T>
using NeighbourGenerator = void(*)(const T& source, T& target, bool& target_exists, uint32_t& cost, unsigned char i);

template<typename T>
using Heuristic = uint32_t(*)(const T&);

template<typename T>
using BinOp = T(*)(const T&, const T&);

template<typename T, typename U>
using UnOp = U(*)(const T&);

template<typename T>
using OrdFun = bool(*)(const T&, const T&);

template<typename T>
using EqFun = bool(*)(const T&, const T&);

template<typename T>
using Hash = uint64_t(*)(const T&);

#endif
