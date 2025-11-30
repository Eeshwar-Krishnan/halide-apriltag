cmake_minimum_required(VERSION 3.16)
project(apriltag_halide LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Aggressive optimization for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math")

# Make sure we're in Release mode
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Halide provides a CMake config package. Set Halide_DIR if CMake cannot find it automatically.
find_package(Halide REQUIRED)

# Find image I/O libraries for Halide::ImageIO
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

add_executable(apriltag_generator src/apriltag_generator.cpp)
target_link_libraries(apriltag_generator PRIVATE Halide::Halide)

set(GENERATED_LIBS
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_edge_detect.a
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_edge_detect.h
)

add_custom_command(
    OUTPUT ${GENERATED_LIBS}
    COMMAND apriltag_generator
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS apriltag_generator
    COMMENT "Generating Halide AOT libraries"
)

add_custom_target(build_apriltag_libs DEPENDS ${GENERATED_LIBS})

add_library(apriltag_edge_detect STATIC IMPORTED GLOBAL)
set_target_properties(apriltag_edge_detect PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/apriltag_edge_detect.a)
add_dependencies(apriltag_edge_detect build_apriltag_libs)

add_executable(apriltag_runner src/main.cpp)
add_dependencies(apriltag_runner build_apriltag_libs)
target_include_directories(apriltag_runner PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

find_library(METAL_LIBRARY Metal)
find_library(FOUNDATION_LIBRARY Foundation)

# Link against the Halide runtime, generated pipelines, and image IO helpers.
# Link libraries directly to avoid duplicate symbol issues
target_link_libraries(apriltag_runner PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_edge_detect.a
    Halide::Runtime
    Halide::ImageIO
    PNG::PNG
    JPEG::JPEG
    ${METAL_LIBRARY}
    ${FOUNDATION_LIBRARY}
)

# Add linker flags to handle duplicate symbols
target_link_options(apriltag_runner PRIVATE
    "-Wl,-dead_strip"
)

