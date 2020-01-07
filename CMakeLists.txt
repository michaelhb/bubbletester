cmake_minimum_required(VERSION 2.8.12)

project(libbubbler_example)
enable_language(CXX)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# Find & set up BP (old style)
find_package(libbubbler REQUIRED)

# Find & set up effpotential (new style)
find_package(effpotential REQUIRED)

# Check we can link everything to the test executable
add_executable(test_linkage src/test_linkage.cpp)
target_link_libraries(test_linkage ${LIBBUBBLER_LIBRARIES})
target_include_directories(test_linkage PUBLIC ${LIBBUBBLER_INCLUDE_DIRS})
target_include_directories(test_linkage PUBLIC $<INSTALL_INTERFACE:include/libeffpotential>)
set_target_properties(test_linkage PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)