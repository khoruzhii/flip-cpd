cmake_minimum_required(VERSION 3.10)

project(flip_cpd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
)

find_package(Threads REQUIRED)

function(add_flip_executable base)
    add_executable(${base}2 src/${base}.cpp)
    target_link_libraries(${base}2 PRIVATE Threads::Threads)

    add_executable(${base}3 src/${base}.cpp)
    target_compile_definitions(${base}3 PRIVATE MOD3)
    target_link_libraries(${base}3 PRIVATE Threads::Threads)
endfunction()

foreach(tool search lift select select_modular)
    add_flip_executable(${tool})
endforeach()
