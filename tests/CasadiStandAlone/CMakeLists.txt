add_executable(CasadiMaupertuis CasadiMaupertuis.cpp)
target_link_libraries(CasadiMaupertuis ${CASADI_LIBRARIES})
target_include_directories(CasadiMaupertuis PUBLIC ${CASADI_INCLUDE_DIR})
set_target_properties(CasadiMaupertuis PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)