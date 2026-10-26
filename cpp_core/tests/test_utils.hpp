#pragma once

#include <cstdlib>
#include <iostream>

// Non viene disabilitato da NDEBUG, quindi i controlli restano attivi anche in Release.
#define GDM_CHECK(condition)                                                        \
    do {                                                                            \
        if (!(condition)) {                                                        \
            std::cerr << "CHECK FAILED: " #condition << " at " << __FILE__ << ':' \
                      << __LINE__ << std::endl;                                     \
            std::exit(EXIT_FAILURE);                                                \
        }                                                                           \
    } while (false)
