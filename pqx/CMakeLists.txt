find_package(Boost CONFIG REQUIRED )
find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

add_library(pqx INTERFACE)

add_library(pqx::pqx ALIAS pqx)

target_sources(pqx PUBLIC FILE_SET HEADERS
				FILES
                parquet.hpp
				type_name.hpp
			   )

target_link_libraries(pqx
		INTERFACE 
		Boost::boost
        Arrow::arrow$<IF:$<BOOL:${ARROW_LINK_STATICALLY}>,_static,_shared>
        Parquet::parquet$<IF:$<BOOL:${ARROW_LINK_STATICALLY}>,_static,_shared>
)

target_include_directories(pqx INTERFACE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
		)

include(GNUInstallDirs)

install(TARGETS pqx
        EXPORT pqx-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pqx
)

install(EXPORT pqx-targets
		FILE pqx-targets.cmake
		NAMESPACE pqx::
		DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}
)




