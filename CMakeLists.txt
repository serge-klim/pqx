cmake_minimum_required(VERSION 3.24)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()


project(pqx VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option (ARROW_LINK_STATICALLY "link Arrow statically" OFF)
add_subdirectory(pqx)


include(CMakePackageConfigHelpers)

write_basic_package_version_file(
		${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
		VERSION ${CMAKE_PROJECT_VERSION}
		COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/config.cmake.in
							${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
							INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

include(GNUInstallDirs)
install(FILES 
			${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
		DESTINATION
            ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}
)

