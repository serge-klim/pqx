#pragma once
#include <boost/type_index.hpp>
#include <string>

namespace pqx { inline namespace v1 {

template<typename T>
struct type_name {
	std::string operator()() {
		return boost::typeindex::type_id<T>().pretty_name();
	}
};

} }


