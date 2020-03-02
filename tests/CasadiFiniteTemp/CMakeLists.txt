add_executable(CasadiFiniteTemp CasadiFiniteTemp.cpp ${SRC_FILES})
target_link_libraries(CasadiFiniteTemp effpotential ${LIBBUBBLER_LIBRARIES} ${CASADI_LIBRARIES})
target_include_directories(CasadiFiniteTemp PUBLIC ${LIBBUBBLER_INCLUDE_DIRS} ${CASADI_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include)
set_target_properties(CasadiFiniteTemp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)