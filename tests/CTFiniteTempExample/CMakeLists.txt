add_executable(CTFiniteTempExample TestSimple2DModel.cpp Simple2DModel.cpp ${SRC_FILES})
target_link_libraries(CTFiniteTempExample ${LIBBUBBLER_LIBRARIES} effpotential)
target_include_directories(CTFiniteTempExample PUBLIC ${LIBBUBBLER_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
set_target_properties(CTFiniteTempExample PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)