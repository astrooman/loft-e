#ifndef _H_LOFTE_PRINT_SAFE
#define _H_LOFTE_PRINT_SAFE

#include <iostream>
#include <mutex>
#include <string>

std::recursive_mutex coutmutex;

template <class T>
void PrintSafe(T lastin) {
    std::lock_guard<std::recursive_mutex> coutlock(coutmutex);
    std::cout << lastin << std::endl;
}

template <class T, class ... Types>
void PrintSafe(T firstin, Types ... args) {
    std::lock_guard<std::recursive_mutex> coutlock(coutmutex);
    std::cout << firstin << " ";
    PrintSafe(args...);
}

#endif
