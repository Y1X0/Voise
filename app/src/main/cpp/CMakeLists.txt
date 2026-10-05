cmake_minimum_required(VERSION 3.18.1)
project(voiceanon_jni CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Real-time DSP must be optimised even in debug APKs.
set(CMAKE_CXX_FLAGS_DEBUG "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# Shared, platform-independent DSP core (the same code the host tests run).
set(VOICEANON_BUILD_HOST_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../dsp ${CMAKE_CURRENT_BINARY_DIR}/dsp)

find_package(oboe REQUIRED CONFIG)

add_library(voiceanon_jni SHARED
  duplex_engine.cpp
  jni_bridge.cpp)
target_compile_options(voiceanon_jni PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(voiceanon_jni voiceanon_dsp oboe::oboe log)
