cmake_minimum_required(VERSION 3.16)

project(eigvecs
    DESCRIPTION "Robust level-3 BLAS Inverse Iteration from the Hessenberg Matrix"
    LANGUAGES C)

# Set C standard to gnu99
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

set(CMAKE_C_FLAGS "-Wall")

set(CMAKE_VERBOSE_MAKEFILE ON)

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -funroll-loops -fprefetch-loop-arrays -LNO:prefetch")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -funroll-loops")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "IntelLLVM")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
else()
    message(WARNING "Untested compiler.")
endif()

add_compile_definitions(ALIGNMENT=64)

include(CheckCCompilerFlag)
CHECK_C_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "IntelLLVM")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=xHost")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    endif()
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(eigvecs inverse_iteration.c main.c partition.c problemgenerator.c
    reduce.c robust.c shifted_Hessenberg_solve.c solve-task.c update-task.c utils.c
    validation.c)
target_link_libraries(eigvecs ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OpenMP_C_LIBRARIES} m)