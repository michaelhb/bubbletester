add_executable(CTFiniteTempExample CTFiniteTempExample.cpp ${SRC_FILES})
target_link_libraries(CTFiniteTempExample effpotential ${CASADI_LIBRARIES} ${LIBBUBBLER_LIBRARIES})
target_include_directories(CTFiniteTempExample PUBLIC ${LIBBUBBLER_INCLUDE_DIRS} ${CASADI_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include)
set_target_properties(CTFiniteTempExample PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)