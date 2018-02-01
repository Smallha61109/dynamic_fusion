cmake_minimum_required(VERSION 3.5.0)

# ---[ Configurations types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Possible configurations" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -DOM_STATIC_BUILD -std=c++11 -DFORCE_INLINERS -D_MWAITXINTRIN_H_INCLUDED -Wno-write-strings")
set(CMAKE_CXX_STANDARD 11)
if (DEFINED CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)

    set(DOXYGEN_INPUT ${SOURCE_FILES})
    set(DOXYGEN_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    add_custom_command(
            OUTPUT ${DOXYGEN_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E echo_append "Building API Documentation..."
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
            COMMAND ${CMAKE_COMMAND} -E echo "Done."
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS ${DOXYGEN_INPUT}
    )

    add_custom_target(apidoc ALL DEPENDS ${DOXYGEN_OUTPUT})

    add_custom_target(apidoc_forced
            COMMAND ${CMAKE_COMMAND} -E echo_append "Building API Documentation..."
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
            COMMAND ${CMAKE_COMMAND} -E echo "Done."
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )

  endif(DOXYGEN_FOUND)
endif()

# ---[ Solution name
project(kfusion C CXX)

# ---[ utility
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")
include(cmake/Utils.cmake)
include(cmake/Targets.cmake)

# ---[ find dependencies
find_package(OpenCV REQUIRED COMPONENTS core viz highgui calib3d)
find_package(CUDA REQUIRED)
find_package(OpenNI)
find_package(Ceres REQUIRED)
# find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${OPENNI_INCLUDE_DIR} "kfusion/src/utils" "kfusion/include/nanoflann")
if(OPENNI_FOUND)
    message("FOUND OPENNI AT: ${OPENNI_INCLUDE_DIR}")
endif()
# ---[ misc settings
if(USE_PROJECT_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# ---[ cuda settings
set(HAVE_CUDA 1)
list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_61,code=sm_61")
#
#if(UNIX OR APPLE)
#  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC;")
#endif()

warnings_disable(CMAKE_CXX_FLAGS /wd4985)

add_subdirectory(kfusion)
add_subdirectory(apps)

set(BUILD_TESTS OFF)
find_path(MLIB_INCLUDE_PATH NAMES include PATHS ~/library/Opt/examples/external/mLib/ NO_DEFAULT_PATH)
find_path(OPT_INCLUDE_PATH NAMES shared/cudaUtil.h PATHS ~/library/Opt/examples/ NO_DEFAULT_PATH)
find_path(OPTAPI_INCLUDE_PATH NAMES include PATHS ~/library/Opt/API/release NO_DEFAULT_PATH)
if(MLIB_INCLUDE_PATH AND OPT_INCLUDE_PATH AND OPTAPI_INCLUDE_PATH)
  set(OPT_INCLUDE_DIRS ${MLIB_INCLUDE_PATH}/include ${OPT_INCLUDE_PATH}/shared ${OPTAPI_INCLUDE_PATH}/include)
  message("-- OPT_INCLUDE_DIRS found: ${OPT_INCLUDE_DIRS}")
endif()
find_library(OPT_LIBRARIES NAMES libOptDev.a PATHS ~/library/Opt/API/release/lib)
if(OPT_LIBRARIES)
  message("-- OPT_LIBRARIES found: ${OPT_LIBRARIES}")
endif()
find_library(TERRA_LIBRARIES NAMES libterra.a PATHS ~/library/terra/lib)
if(TERRA_LIBRARIES)
  message("-- TERRA_LIBRARIES found: ${TERRA_LIBRARIES}")
endif()
find_library(OPENMESH_CORE_LIBRARY NAMES libOpenMeshCore.a PATHS ~/library/Opt/examples/external/OpenMesh/lib/linux)
find_library(OPENMESH_TOOL_LIBRARY NAMES libOpenMeshTools.a PATHS ~/library/Opt/examples/external/OpenMesh/lib/linux)
if(OPENMESH_CORE_LIBRARY AND OPENMESH_TOOL_LIBRARY)
  set(OPENMESH_LIBRARIES ${OPENMESH_CORE_LIBRARY} ${OPENMESH_TOOL_LIBRARY})
  message("-- OPENMESH_LIBRARIES found: ${OPENMESH_LIBRARIES}")
endif()
find_path(OPENMESH_INCLUDE_DIR NAMES include PATHS ~/library/Opt/examples/external/OpenMesh NO_DEFAULT_PATH)
if(OPENMESH_INCLUDE_DIR)
  set(OPENMESH_INCLUDE_DIRS ${OPENMESH_INCLUDE_DIR}/include)
  message("-- OPENMESH_INCLUDE_DIRS found: ${OPENMESH_INCLUDE_DIRS}")
endif()
if(OPT_INCLUDE_DIRS AND OPT_LIBRARIES AND TERRA_LIBRARIES AND OPENMESH_LIBRARIES AND OPENMESH_INCLUDE_DIRS)
  set(BUILD_TESTS ON)
  message("-- All dependencies satisfied, BUILD_TESTS set to ON")
else()
  message("-- Warning: BUILD_TESTS set to OFF due to missing dependecy")
endif()

# set(OPT_INCLUDE_DIRS /home/mml-taitanx/library/Opt/examples/external/mLib/include /home/mml-taitanx/library/Opt/examples/shared/ /home/mml-taitanx/library/Projects/Opt/API/release/include)
# set(OPT_HEADER_PATH /home/mml-taitanx/library/Opt/API/release/include)
# set(OPT_LIBRARIES /home/mml-taitanx/library/Opt/API/release/lib/libOptDev.a)
# set(TERRA_LIBRARIES /home/mml-taitanx/library/terra/lib/libterra.a)
# set(OPENMESH_LIBRARIES /home/mml-taitanx/library/Opt/examples/external/OpenMesh/lib/linux/libOpenMeshCore.a /home/mml-taitanx/library/Opt/examples/external/OpenMesh/lib/linux/libOpenMeshTools.a)
# set(OPENMESH_INCLUDE_DIRS /home/mml-taitanx/library/Opt/examples/external/OpenMesh/include)

if(BUILD_TESTS)
  find_package(GTest REQUIRED)
  if( GTEST_FOUND )
    message( "Found Gtest at ${GTEST_ROOT}")
    message( "GTest Libs: ${GTEST_BOTH_LIBRARIES}")
    message( "GTest Include: ${GTEST_INCLUDE_DIRS}")
    include_directories(${GTEST_INCLUDE_DIRS})
    add_subdirectory(tests)
  endif()
endif()
