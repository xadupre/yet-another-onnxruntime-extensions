
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
  cmake_policy(SET CMP0077 NEW)
endif()

#
# initialisation
#

message(STATUS "--------------------------------------------")
message(STATUS "CMAKE_VERSION=${CMAKE_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_C_COMPILER_VERSION=${CMAKE_C_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "ORT_VERSION=${ORT_VERSION}")
message(STATUS "--------------------------------------------")

#
# platform extension
#
if(MSVC)
  set(DLLEXT "dll")
elseif(APPLE)
  set(DLLEXT "dylib")
else()
  set(DLLEXT "so")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")

if(APPLE)
  set(DEFAULT_OSX_DEPLOYMENT_TARGET "10.15")
  set(CMAKE_OSX_DEPLOYMENT_TARGET "${DEFAULT_OSX_DEPLOYMENT_TARGET}")
endif()

#
# C++ standard
#
if(MSVC)
  set(CMAKE_CXX_STANDARD 20)
elseif(APPLE)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wextra")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unknown-pragmas -Wextra")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wextra")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unknown-pragmas -Wextra")
  if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "15")
    set(CMAKE_CXX_STANDARD 23)
  elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
    set(CMAKE_CXX_STANDARD 20)
  elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "9")
    set(CMAKE_CXX_STANDARD 17)
  else()
    message(FATAL_ERROR "gcc>=9 is required but ${CMAKE_C_COMPILER_VERSION} was detected.")
  endif()
endif()

#
# Compiler options
#
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
  add_compile_options(/wd4068)
else()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -mf16c")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

message(STATUS "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}")

#
# Code coverage
#
option(ENABLE_COVERAGE "Enable C/C++ code coverage instrumentation" OFF)
if(ENABLE_COVERAGE)
  if(MSVC)
    message(WARNING "Code coverage is not supported with MSVC; ENABLE_COVERAGE is ignored.")
  else()
    message(STATUS "Code coverage is enabled.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
  endif()
endif()
message(STATUS "ENABLE_COVERAGE=${ENABLE_COVERAGE}")
