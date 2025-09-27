cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(aoti_example)

find_package(Torch REQUIRED)

# Build the test executable
add_executable(aoti_example compress_test.cpp)

# Make sure the CAESAR library is visible
# Adjust the path if CAESAR_C is not at this relative location
add_subdirectory(../../CAESAR_C/CAESAR ${CMAKE_BINARY_DIR}/caesar_build)

# Link against both Torch and caesar_lib
target_link_libraries(aoti_example
    PRIVATE
        "${TORCH_LIBRARIES}"
        caesar_lib
)

set_property(TARGET aoti_example PROPERTY CXX_STANDARD 17)

# Custom command to build model.pt2
add_custom_command(
    OUTPUT model.pt2
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/model.py
    DEPENDS model.py
)

