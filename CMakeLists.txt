cmake_minimum_required(VERSION 3.14)
project(cldiff LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies, fetched at configure time ---
include(FetchContent)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
# Eigen ships its own CMakeLists.txt that builds tests/docs we don't want;
# these two lines skip that and just make the headers available.
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.0
)
FetchContent_MakeAvailable(catch2)

# --- Your library ---
add_library(cldiff
    src/polyhedron.cpp
    src/rng.cpp
    src/sampler.cpp
    src/target.cpp
)
target_include_directories(cldiff PUBLIC include)
target_link_libraries(cldiff PUBLIC Eigen3::Eigen)

# Warnings + sanitizers, as discussed early on
target_compile_options(cldiff PRIVATE -Wall -Wextra -Werror)

# --- Tests ---
#enable_testing()
#add_subdirectory(tests)

add_subdirectory(examples)
