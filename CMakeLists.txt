cmake_minimum_required(VERSION 2.8.12)

project(libbubbler_example)
enable_language(CXX)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# Find & set up BP (old style)
find_package(libbubbler REQUIRED)

# Find & set up effpotential (new style)
find_package(effpotential REQUIRED)

# List source files so tests can access them (surely there's a better way?)
set(SRC_FILES ${PROJECT_SOURCE_DIR}/src/SimpleBounce.cpp)

# Set up the tests
add_subdirectory(tests)

