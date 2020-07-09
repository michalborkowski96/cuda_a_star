# CUDA A*

This is an implementation of the parallel A\* algorithm for CUDA, as per the [Massively Parallel A* Search on a GPU paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9620/9366). It is used for solving sliding puzzle (get from initial position to target position) and pathfinding (find the shortest path on a grid with weights).

Main points of the project:

* `bitonic_sort.h` - an implementation of bitonic sort in CUDA
* `device_hashmap.h` - a simple hashmap on a GPU, with almost no locking mechanisms
* `heaps.h` - a class that represents N heaps, allowing parallel operations on them
* `fancy_hashmap.h` - a fancy hashmap on a GPU. This hashmap supports mass insert, where a number of elements is inserted in parallel
* `reduction.h` - a simple implementation of reduction
* `a_star.h` - the algorithm itself
