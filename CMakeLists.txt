cmake_minimum_required(VERSION 3.16)
project(pose_inference VERSION 1.0.0 LANGUAGES CXX)

option(DEBUG_TIME_LOGGING "Enable cout for logging runtime" ON)
option(POSE_INFERENCE_BUILD_EXECUTABLES "Build pose inference applications" ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(cmake/ccache.cmake)

set(POSE_INFERENCE_OPTIMIZATION_FLAGS "$ENV{CPP_OPTIMIZATIONS}" CACHE STRING
  "Additional optimization flags for pose_inference")

set(_POSE_INFERENCE_DEFAULT_BATCH_SIZE 5)
if(DEFINED ENV{BATCH_SIZE})
  set(_POSE_INFERENCE_DEFAULT_BATCH_SIZE "$ENV{BATCH_SIZE}")
endif()
set(BATCH_SIZE "${_POSE_INFERENCE_DEFAULT_BATCH_SIZE}" CACHE STRING
  "Fixed batch size for pose inference")
unset(_POSE_INFERENCE_DEFAULT_BATCH_SIZE)

message(STATUS "DEBUG_TIME_LOGGING is ${DEBUG_TIME_LOGGING}")
message(STATUS "Pose inference batch size is ${BATCH_SIZE}")

set(CONFIG_DIR "${CMAKE_INSTALL_FULL_DATADIR}/${PROJECT_NAME}")
set(POSE_INFERENCE_GENERATED_INCLUDE_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
file(MAKE_DIRECTORY
  "${POSE_INFERENCE_GENERATED_INCLUDE_DIR}/${PROJECT_NAME}")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in
  ${POSE_INFERENCE_GENERATED_INCLUDE_DIR}/${PROJECT_NAME}/config.h
  @ONLY
)

set(OPENCV_VERSION "$ENV{OPENCV_VERSION}" CACHE STRING
  "Minimum OpenCV version; empty accepts any compatible version")
if(POLICY CMP0146)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0146 OLD)
endif()
if(OPENCV_VERSION)
  find_package(OpenCV ${OPENCV_VERSION} REQUIRED COMPONENTS
    core dnn imgproc imgcodecs videoio cudaarithm cudaimgproc cudawarping)
else()
  find_package(OpenCV REQUIRED COMPONENTS
    core dnn imgproc imgcodecs videoio cudaarithm cudaimgproc cudawarping)
endif()
if(POLICY CMP0146)
  cmake_policy(POP)
endif()

find_package(Threads REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(cpp_utils CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(tensorrt-cpp-api CONFIG REQUIRED)

# ============ LIBRARY ============
add_library(${PROJECT_NAME} STATIC
  src/config_parser.cpp
  src/detection.cpp
  src/stages.cpp
  src/utils.cpp
)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

if(POSE_INFERENCE_OPTIMIZATION_FLAGS)
  separate_arguments(POSE_INFERENCE_OPTIMIZATION_OPTIONS NATIVE_COMMAND
    "${POSE_INFERENCE_OPTIMIZATION_FLAGS}")
  target_compile_options(${PROJECT_NAME} PRIVATE
    ${POSE_INFERENCE_OPTIMIZATION_OPTIONS})
endif()

if(DEBUG_TIME_LOGGING)
  target_compile_definitions(${PROJECT_NAME} PUBLIC USE_DEBUG_TIME_LOGGING=1)
endif()

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${POSE_INFERENCE_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${OpenCV_LIBS}
    Threads::Threads
    Eigen3::Eigen
    cpp_utils::cpp_utils
    tensorrt-cpp-api
)

# ============ EXECUTABLES ============

if(POSE_INFERENCE_BUILD_EXECUTABLES)
  add_executable(inference_benchmark src/apps/inference_benchmark.cpp)
  target_link_libraries(inference_benchmark PRIVATE
    ${PROJECT_NAME}::${PROJECT_NAME})

  add_executable(video_inference_export src/apps/video_inference_export.cpp)
  target_link_libraries(video_inference_export PRIVATE
    ${PROJECT_NAME}::${PROJECT_NAME})

  add_executable(keypoint_overlay src/apps/keypoint_overlay.cpp)
  target_link_libraries(keypoint_overlay PRIVATE
    ${PROJECT_NAME}::${PROJECT_NAME})
endif()

# ============ INSTALLATION ============

set(POSE_INFERENCE_INSTALL_CMAKEDIR
  "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${POSE_INFERENCE_INSTALL_CMAKEDIR}
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${POSE_INFERENCE_INSTALL_CMAKEDIR}
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

export(EXPORT ${PROJECT_NAME}Targets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
  NAMESPACE ${PROJECT_NAME}::
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${POSE_INFERENCE_INSTALL_CMAKEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
install(FILES
  "${POSE_INFERENCE_GENERATED_INCLUDE_DIR}/${PROJECT_NAME}/config.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
install(DIRECTORY cfg/ DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}"
  FILES_MATCHING PATTERN "*.json")
