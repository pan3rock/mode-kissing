cmake_minimum_required(VERSION 3.10.0)
project(mode-kissing VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-non-virtual-dtor")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -fPIC -ffp-contract=fast")

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

# fmt
add_subdirectory(${CMAKE_SOURCE_DIR}/external/fmt EXCLUDE_FROM_ALL)

# HighFive
set(HIGHFIVE_USE_BOOST OFF)
set(HIGHFIVE_BUILD_DOCS OFF)
set(HIGHFIVE_UNIT_TESTS OFF)
set(HIGHFIVE_EXAMPLES OFF)
add_subdirectory(${CMAKE_SOURCE_DIR}/external/HighFive)

# toml11
add_subdirectory(${CMAKE_SOURCE_DIR}/external/toml11 EXCLUDE_FROM_ALL)

include_directories(${CMAKE_SOURCE_DIR}/include)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(src)
