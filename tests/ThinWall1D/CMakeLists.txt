add_executable(thin_wall_1D ThinWall1D/ThinWall1D.cpp ${SRC_FILES})
target_link_libraries(thin_wall_1D ${LIBBUBBLER_LIBRARIES})
target_include_directories(thin_wall_1D PUBLIC ${LIBBUBBLER_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
set_target_properties(thin_wall_1D PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)