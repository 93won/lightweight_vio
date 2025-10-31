cmake_minimum_required(VERSION 3.10)
project(lightweight_vio)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenGL preference to avoid warnings
set(OpenGL_GL_PREFERENCE GLVND)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Threads REQUIRED)

# Build Pangolin from thirdparty
set(BUILD_PANGOLIN_EXAMPLES OFF CACHE BOOL "Build Pangolin examples" FORCE)
set(BUILD_PANGOLIN_TESTS OFF CACHE BOOL "Build Pangolin tests" FORCE)
set(BUILD_PANGOLIN_TOOLS OFF CACHE BOOL "Build Pangolin tools" FORCE)
add_subdirectory(thirdparty/pangolin EXCLUDE_FROM_ALL)

# Build Sophus from thirdparty first with tests disabled
set(BUILD_TESTS OFF CACHE BOOL "Build Sophus tests" FORCE)
set(BUILD_SOPHUS_TESTS OFF CACHE BOOL "Build Sophus tests" FORCE)
set(BUILD_SOPHUS_EXAMPLES OFF CACHE BOOL "Build Sophus examples" FORCE)
add_subdirectory(thirdparty/Sophus)

# Build spdlog from thirdparty
set(SPDLOG_BUILD_EXAMPLES OFF CACHE BOOL "Build spdlog examples" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "Build spdlog tests" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "Build spdlog benchmarks" FORCE)
add_subdirectory(thirdparty/spdlog)

# Build Ceres from thirdparty without finding system version
set(CERES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ceres-solver)
set(BUILD_EXAMPLES OFF CACHE BOOL "Build Ceres examples" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Build Ceres benchmarks" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Build Ceres tests" FORCE)
set(SUITESPARSE OFF CACHE BOOL "Enable SuiteSparse" FORCE)
set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "Disable Ceres uninstall target" FORCE)

# Prevent finding system Ceres
set(CMAKE_DISABLE_FIND_PACKAGE_Ceres ON)

add_subdirectory(${CERES_DIR} EXCLUDE_FROM_ALL)

# Find GLU for legacy OpenGL functions
find_library(GLU_LIBRARY GLU REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(src)
include_directories(thirdparty/pangolin/include)
include_directories(${CMAKE_BINARY_DIR}/thirdparty/pangolin/include)
include_directories(thirdparty/Sophus)
include_directories(thirdparty/spdlog/include)
include_directories(thirdparty/ceres-solver/include)
include_directories(${CMAKE_BINARY_DIR}/thirdparty/ceres-solver/include)

# Collect source files
file(GLOB_RECURSE SOURCES 
    "src/database/*.cpp"
    "src/module/*.cpp"
    "src/util/*.cpp"
    "src/viewer/*.cpp"
)



# Add unified stereo pipeline (VO/VIO configurable via YAML)
add_executable(euroc_stereo app/euroc_stereo.cpp 
    app/player/euroc_player.cpp
    src/camera/Camera.cpp
    src/camera/PinholeCamera.cpp
    src/camera/FisheyeCamera.cpp
    src/database/Frame.cpp
    src/database/Feature.cpp
    src/database/MapPoint.cpp
    src/processing/FeatureTracker.cpp
    src/processing/Optimizer.cpp
    src/processing/Estimator.cpp
    src/processing/IMUHandler.cpp
    src/optimization/Factors.cpp
    src/optimization/Parameters.cpp
    src/util/Config.cpp
    src/util/EurocUtils.cpp
    src/viewer/PangolinViewer.cpp
)

# Add TUM stereo pipeline (Fisheye support)
add_executable(tum_stereo app/tum_stereo.cpp 
    app/player/tum_player.cpp
    src/camera/Camera.cpp
    src/camera/PinholeCamera.cpp
    src/camera/FisheyeCamera.cpp
    src/database/Frame.cpp
    src/database/Feature.cpp
    src/database/MapPoint.cpp
    src/processing/FeatureTracker.cpp
    src/processing/Optimizer.cpp
    src/processing/Estimator.cpp
    src/processing/IMUHandler.cpp
    src/optimization/Factors.cpp
    src/optimization/Parameters.cpp
    src/util/Config.cpp
    src/util/TUMUtils.cpp
    src/viewer/PangolinViewer.cpp
)

# Add RGBD pipeline (VO mode)
add_executable(rgbd_example app/rgbd_example.cpp 
    app/player/rgbd_player.cpp
    src/camera/Camera.cpp
    src/camera/PinholeCamera.cpp
    src/camera/FisheyeCamera.cpp
    src/database/Frame.cpp
    src/database/Feature.cpp
    src/database/MapPoint.cpp
    src/processing/FeatureTracker.cpp
    src/processing/Optimizer.cpp
    src/processing/Estimator.cpp
    src/processing/IMUHandler.cpp
    src/optimization/Factors.cpp
    src/optimization/Parameters.cpp
    src/util/Config.cpp
    src/viewer/PangolinViewer.cpp
)


# Link libraries
target_link_libraries(euroc_stereo 
    ${OpenCV_LIBS} 
    yaml-cpp 
    OpenGL::GL 
    GLEW::GLEW
    Threads::Threads
    ceres
    Sophus::Sophus
    spdlog::spdlog
    pango_display
    pango_windowing
    pango_opengl
    pango_core
    pango_image
    m  # Math library for M_PI
)

target_link_libraries(tum_stereo 
    ${OpenCV_LIBS} 
    yaml-cpp 
    OpenGL::GL 
    GLEW::GLEW
    Threads::Threads
    ceres
    Sophus::Sophus
    spdlog::spdlog
    pango_display
    pango_windowing
    pango_opengl
    pango_core
    pango_image
    m  # Math library for M_PI
)

target_link_libraries(rgbd_example 
    ${OpenCV_LIBS} 
    yaml-cpp 
    OpenGL::GL 
    GLEW::GLEW
    Threads::Threads
    ceres
    Sophus::Sophus
    spdlog::spdlog
    pango_display
    pango_windowing
    pango_opengl
    pango_core
    pango_image
    m  # Math library for M_PI
)

# Link Eigen3 to all targets
if(TARGET Eigen3::Eigen)
    target_link_libraries(euroc_stereo Eigen3::Eigen)
    target_link_libraries(tum_stereo Eigen3::Eigen)
    target_link_libraries(rgbd_example Eigen3::Eigen)
else()
    # Fallback for older CMake versions
    target_include_directories(euroc_stereo PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_include_directories(tum_stereo PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_include_directories(rgbd_example PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

# Print OpenCV information for debugging
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
