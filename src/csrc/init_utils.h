#ifndef INIT_UTILS_H
#define INIT_UTILS_H

#include <random>

inline void random_init(float *array, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> dist(0.0, 1.0);

    for(int i=0; i < size; ++i) {
        array[i] = dist(gen);
    } 
}

#endif // INIT_UTILS_H