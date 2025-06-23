#pragma once

#include "type_name.hpp"
#include "parquet/types.h"
#include "parquet/schema.h"
#include "parquet/file_writer.h"
#include "parquet/column_writer.h"
#include <boost/describe.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
#include <optional>
#include <utility>
#include <limits>
#include <chrono>
#include <array>
#include <string>
#include <type_traits>
#include <cstdint>

namespace pqx {
inline namespace v1 {

template <std::size_t Level>
struct flatten_to {
   static constexpr std::size_t level = Level;
};

using flatten = flatten_to<std::numeric_limits<std::size_t>::max()>;

namespace detail {

template <typename T>
struct is_flatten_to : std::false_type {};

template <std::size_t Level>
struct is_flatten_to<flatten_to<Level>> : std::true_type {};

} // namespace detail

template <typename... O>
using options = boost::mp11::mp_list<O...>;

namespace traits {

template <std::size_t N>
struct int_t;

template <std::size_t N>
struct uint_t;

template <>
struct int_t<sizeof(std::int64_t)> {
   static constexpr parquet::Type::type type = parquet::Type::INT64;
   static constexpr parquet::ConvertedType::type converted_type = parquet::ConvertedType::INT_64;
};

template <>
struct uint_t<sizeof(std::uint64_t)> {
   static constexpr parquet::Type::type type = parquet::Type::INT64;
   static constexpr parquet::ConvertedType::type converted_type = parquet::ConvertedType::UINT_64;
};

template <>
struct int_t<sizeof(std::int32_t)> {
   static constexpr parquet::Type::type type = parquet::Type::INT32;
   static constexpr parquet::ConvertedType::type converted_type = parquet::ConvertedType::INT_32;
};

template <>
struct uint_t<sizeof(std::uint32_t)> {
   static constexpr parquet::Type::type type = parquet::Type::INT32;
   static constexpr parquet::ConvertedType::type converted_type = parquet::ConvertedType::UINT_32;
};

template <>
struct int_t<sizeof(std::int16_t)> {
   static constexpr parquet::Type::type type = parquet::Type::INT32;
   static constexpr parquet::ConvertedType::type converted_type = parquet::ConvertedType::INT_16;
};

template <>
struct uint_t<sizeof(std::uint16_t)> {
   static constexpr parquet::Type::type type = parquet::Type::INT32;
   static constexpr parquet::ConvertedType::type converted_type = parquet::ConvertedType::UINT_16;
};

template <>
struct int_t<sizeof(std::int8_t)> {
   static constexpr parquet::Type::type type = parquet::Type::INT32;
   static constexpr parquet::ConvertedType::type converted_type = parquet::ConvertedType::INT_8;
};

template <>
struct uint_t<sizeof(std::uint8_t)> {
   static constexpr parquet::Type::type type = parquet::Type::INT32;
   static constexpr parquet::ConvertedType::type converted_type = parquet::ConvertedType::UINT_8;
};

// template <typename Rep, typename Options>
// struct duration_t<std::chrono::duration<Rep, std::milli>, typename std::is_integral<Rep>::type> {
//    static constexpr parquet::Type::type type = parquet::Type::INT64;
//    static constexpr parquet::ConvertedType::type converted_type = parquet::ConvertedType::TIMESTAMP_MILLIS;
// };
//
// template <typename Rep, typename Options>
// struct duration_t<std::chrono::duration<Rep, std::micro>, typename std::is_integral<Rep>::type> {
//    static constexpr parquet::Type::type type = parquet::Type::INT64;
//    static constexpr parquet::ConvertedType::type converted_type = parquet::ConvertedType::TIMESTAMP_MICROS;
// };
//
// template <typename Rep, typename Period, typename Options>
// struct duration_t<std::chrono::duration<Rep, Period>, std::conjunction<std::is_integral<Rep>, std::ratio_greater_equal<std::micro, Period>>> {
//    static constexpr parquet::Type::type type = parquet::Type::INT64;
//    static constexpr parquet::ConvertedType::type converted_type = parquet::ConvertedType::TIMESTAMP_MICROS;
// };

} // namespace traits

namespace detail {

template <auto M>
struct member_type;
template <typename T, typename U, U T::* P>
struct member_type<P> {
   using type = U;
};

template <typename T>
void write_one(parquet::TypedColumnWriter<T>* writer, typename T::c_type value);

template <typename T>
void write_one(parquet::TypedColumnWriter<T>* writer, std::nullptr_t);

template <typename T>
struct is_repetition : std::false_type {};

template <parquet::Repetition::type R>
struct is_repetition<std::integral_constant<parquet::Repetition::type, R>> : std::true_type {};

template <typename T, parquet::Repetition::type DefaultRepetitionType>
struct select_repetition;

template <template <class...> class L, parquet::Repetition::type R, parquet::Repetition::type DefaultRepetitionType>
struct select_repetition<L<std::integral_constant<parquet::Repetition::type, R>>, DefaultRepetitionType> : std::integral_constant<parquet::Repetition::type, R> {};

template <template <class...> class L, parquet::Repetition::type DefaultRepetitionType>
struct select_repetition<L<>, DefaultRepetitionType> : std::integral_constant<parquet::Repetition::type, DefaultRepetitionType> {};

template <typename Options, parquet::Repetition::type DefaultRepetitionType = parquet::Repetition::REQUIRED>
static constexpr parquet::Repetition::type repetition_from_options =
    select_repetition<boost::mp11::mp_copy_if<Options, is_repetition>, DefaultRepetitionType>::value;

template <int N, typename Options, parquet::ConvertedType::type ConvertedType = parquet::ConvertedType::NONE>
struct fixed_len_byte_array_node {
   parquet::schema::NodePtr operator()(std::string const& name) const {
      return parquet::schema::PrimitiveNode::Make(name, repetition_from_options<Options>,
                                                  parquet::Type::FIXED_LEN_BYTE_ARRAY,
                                                  ConvertedType, N);
   }
};

} // namespace detail

template <typename T, typename Options, typename Enable = std::true_type>
struct node_builder;

struct skip;

struct node_skipper {
   constexpr skip operator()(std::string const& name) const noexcept;
};

template <typename Options>
struct node_builder<char, Options, std::true_type> : detail::fixed_len_byte_array_node<1, Options> {
};

template <std::size_t N, typename Options>
struct node_builder<std::array<char, N>, Options, std::true_type> : detail::fixed_len_byte_array_node<N, Options> {
};

template <typename T, typename Options>
struct node_builder<T, Options, typename std::is_integral<T>::type> {
   parquet::schema::NodePtr operator()(std::string const& name) const {
      using traits = std::conditional_t<std::is_signed_v<T>, traits::int_t<sizeof(T)>, traits::uint_t<sizeof(T)>>;
      return parquet::schema::PrimitiveNode::Make(name, detail::repetition_from_options<Options>, traits::type, traits::converted_type);
   }
};

// template <typename Rep, typename Options>
// struct node_builder<std::chrono::duration<Rep, std::nano>, Options, typename std::is_integral<Rep>::type> {
//    parquet::schema::NodePtr operator()(std::string const& name) const {
//       return parquet::schema::PrimitiveNode::Make(name, detail::repetition_from_options<Options>, parquet::Type::INT96);
//    }
// };

// template <typename Rep, typename Options>
// struct node_builder<std::chrono::duration<Rep, std::micro>, Options, typename std::is_integral<Rep>::type> {
//    parquet::schema::NodePtr operator()(std::string const& name) const {
//       return parquet::schema::PrimitiveNode::Make(name, detail::repetition_from_options<Options>, parquet::Type::INT64, parquet::ConvertedType::TIMESTAMP_MICROS);
//    }
// };

// template <typename Rep, typename Period, typename Options>
// struct node_builder<std::chrono::duration<Rep, Period>, Options, typename std::conjunction<std::is_integral<Rep>, std::ratio_greater_equal<Period, std::milli>>::type> {
//    parquet::schema::NodePtr operator()(std::string const& name) const {
//       return parquet::schema::PrimitiveNode::Make(name, detail::repetition_from_options<Options>, parquet::Type::INT64, parquet::ConvertedType::TIMESTAMP_MILLIS);
//    }
// };

template <typename Clock, typename Rep, typename Options>
struct node_builder<std::chrono::time_point<Clock, std::chrono::duration<Rep, std::nano>>, Options, typename std::is_integral<Rep>::type> {
   parquet::schema::NodePtr operator()(std::string const& name) const {
      return parquet::schema::PrimitiveNode::Make(name, detail::repetition_from_options<Options>, parquet::Type::INT64, parquet::ConvertedType::TIMESTAMP_MICROS);
   }
};

template <typename Clock, typename Rep, typename Options>
struct node_builder<std::chrono::time_point<Clock, std::chrono::duration<Rep, std::micro>>, Options, typename std::is_integral<Rep>::type> {
   parquet::schema::NodePtr operator()(std::string const& name) const {
      return parquet::schema::PrimitiveNode::Make(name, detail::repetition_from_options<Options>, parquet::Type::INT64, parquet::ConvertedType::TIMESTAMP_MILLIS);
   }
};

template <typename Clock, typename Rep, typename Period, typename Options>
struct node_builder<std::chrono::time_point<Clock, std::chrono::duration<Rep, Period>>, Options, typename std::conjunction<std::is_integral<Rep>, std::ratio_greater_equal<Period, std::milli>>::type> {
   parquet::schema::NodePtr operator()(std::string const& name) const {
      return parquet::schema::PrimitiveNode::Make(name, detail::repetition_from_options<Options>, parquet::Type::INT64, parquet::ConvertedType::TIMESTAMP_MILLIS);
   }
};

template <typename T, typename Options>
struct node_builder<T, Options, typename std::is_enum<T>::type> {
   parquet::schema::NodePtr operator()(std::string const& name) const {
      return node_builder<std::underlying_type_t<T>, Options>{}(name);
   }
};

template <typename T, typename Options>
struct node_builder<std::optional<T>, Options, std::true_type> {
   parquet::schema::NodePtr operator()(std::string const& name) const {
      using extended_options = boost::mp11::mp_push_back<Options,
                                                         std::integral_constant<parquet::Repetition::type, parquet::Repetition::OPTIONAL>>;
      return node_builder<T, extended_options>{}(name);
   }
};

template <typename T, typename Options>
void add_fields(parquet::schema::NodeVector& fields) {
   using public_members = boost::describe::describe_members<T, boost::describe::mod_public>;
   boost::mp11::mp_for_each<public_members>([&fields](auto D) {
      using member_type = typename detail::member_type<decltype(D)::pointer>::type;
      if constexpr (!std::is_same_v<decltype(std::declval<node_builder<member_type, Options>&>()(D.name)), skip>) {
         static constexpr auto flatten_to_level = boost::mp11::mp_front<Options>::level;
         if constexpr (flatten_to_level != 0 && boost::describe::has_describe_members<member_type>::value)
            add_fields<member_type, boost::mp11::mp_replace_front<Options, flatten_to<flatten_to_level - 1>>>(fields);
         else
            fields.push_back(node_builder<member_type, Options>{}(D.name));
      }
   });
}

template <typename... T, typename Options>
struct node_builder<boost::mp11::mp_list<T...>, Options, std::true_type> {
   parquet::schema::NodePtr operator()(std::string const& name) const {
      parquet::schema::NodeVector fields;
      auto i = std::size_t{0};
      (..., push_back<T>(++i, fields));
      return parquet::schema::GroupNode::Make(name, detail::repetition_from_options<Options>, std::move(fields) /*, parquet::LogicalType::List()*/);
   }

 private:
   template <typename U>
   void push_back(std::size_t i, parquet::schema::NodeVector& fields) const {
      if constexpr (!std::is_same_v<decltype(std::declval<node_builder<U, Options>&>()("")), skip>) {
         using O = boost::mp11::mp_partition<Options, detail::is_flatten_to>;
         if constexpr (std::conjunction<std::negation<std::is_same<boost::mp11::mp_first<O>, boost::mp11::mp_list<>>>,
                                        boost::describe::has_describe_members<U>>::value)
            add_fields<U, boost::mp11::mp_flatten<O>>(fields);
         else
            fields.push_back(node_builder<U, Options>{}(std::to_string(1)));
      }
   }
};

template <typename T, typename Options>
struct node_builder<T, Options, typename boost::describe::has_describe_members<T>::type> {
   parquet::schema::NodePtr operator()(std::string const& name) const {
      parquet::schema::NodeVector fields;
      using O = boost::mp11::mp_partition<Options, detail::is_flatten_to>;
      using NormOptions = boost::mp11::mp_push_front<boost::mp11::mp_second<O>,
                                                     std::conditional_t<std::is_same_v<boost::mp11::mp_first<O>, boost::mp11::mp_list<>>,
                                                                        flatten_to<0>,
                                                                        boost::mp11::mp_first<O>>>;
      add_fields<T, NormOptions>(fields);
      return parquet::schema::GroupNode::Make(name, detail::repetition_from_options<Options>, std::move(fields) /*, parquet::LogicalType::List()*/);
   }
};

template <typename T, typename... O>
std::shared_ptr<parquet::schema::GroupNode> make_schema() {
   return std::static_pointer_cast<parquet::schema::GroupNode>(node_builder<T, options<O...>>{}(type_name<T>{}()));
}

template <typename T, typename Enable = std::true_type>
struct column_writer;

class row_group_writer {
 public:
   row_group_writer(parquet::RowGroupWriter* writer) : writer_{writer} {}
   parquet::ColumnWriter* next() {
      return writer_->column(n++);
   }

 private:
   parquet::RowGroupWriter* writer_;
   int n = 0;
};

struct fixed_array_column_writer {
   void operator()(row_group_writer& row_group, parquet::FixedLenByteArray* flba_value) const {
      auto fd_writer = static_cast<parquet::FixedLenByteArrayWriter*>(row_group.next());
      static const auto def_levels = std::int16_t{1};
      static const auto rep_levels = std::int16_t{0};
      fd_writer->WriteBatch(1, &def_levels, &rep_levels, flba_value);
   }
   void operator()(row_group_writer& row_group, std::nullptr_t) const {
      auto fd_writer = static_cast<parquet::FixedLenByteArrayWriter*>(row_group.next());
      static const auto def_levels = std::int16_t{0};
      static const auto rep_levels = std::int16_t{0};
      fd_writer->WriteBatch(1, &def_levels, &rep_levels, nullptr);
   }
};

template <>
struct column_writer<char, std::true_type> : fixed_array_column_writer {
   void operator()(row_group_writer& row_group, char value) const {
      parquet::FixedLenByteArray flba_value;
      flba_value.ptr = reinterpret_cast<uint8_t const*>(&value);
      fixed_array_column_writer::operator()(row_group, &flba_value);
   }
   void operator()(row_group_writer& row_group, std::nullptr_t) const {
      fixed_array_column_writer::operator()(row_group, nullptr);
   }
};

template <std::size_t N>
struct column_writer<std::array<char, N>, std::true_type> : fixed_array_column_writer {
   void operator()(row_group_writer& row_group, std::array<char, N> const& value) const {
      parquet::FixedLenByteArray flba_value;
      flba_value.ptr = reinterpret_cast<uint8_t const*>(value.data());
      fixed_array_column_writer::operator()(row_group, &flba_value);
   }
   void operator()(row_group_writer& row_group, std::nullptr_t) const {
      fixed_array_column_writer::operator()(row_group, nullptr);
   }
};

template <typename T>
struct column_writer<T, typename std::is_integral<T>::type> {
   using traits = std::conditional_t<std::is_signed_v<T>, traits::int_t<sizeof(T)>, traits::uint_t<sizeof(T)>>;
   using parquet_type = parquet::PhysicalType<traits::type>;

   void operator()(row_group_writer& row_group, T value) const {
      detail::write_one(static_cast<parquet::TypedColumnWriter<parquet_type>*>(row_group.next()), static_cast<typename parquet_type::c_type>(value));
   }
   void operator()(row_group_writer& row_group, std::nullptr_t) const {
      detail::write_one(static_cast<parquet::TypedColumnWriter<parquet_type>*>(row_group.next()), nullptr);
   }
};

template <typename T>
struct column_writer<T, typename std::is_enum<T>::type> {
   void operator()(row_group_writer& row_group, T value) const {
      pqx::column_writer<std::underlying_type_t<T>>{}(row_group, static_cast<std::underlying_type_t<T>>(value));
   }
   void operator()(row_group_writer& row_group, std::nullptr_t) const {
      pqx::column_writer<std::underlying_type_t<T>>{}(row_group, nullptr);
   }
};

// template <typename Rep>
// struct column_writer<std::chrono::duration<Rep, std::nano>, typename std::is_integral<Rep>::type> {
//    void operator()(row_group_writer& row_group, std::chrono::duration<Rep, std::nano> const& value) const {
//       auto blob = parquet::Int96{};
//       // TODO:ditch internal function
//       parquet::internal::NanosecondsToImpalaTimestamp(static_cast<std::int64_t>(value.count()), &blob);
//       detail::write_one(static_cast<parquet::TypedColumnWriter<parquet::Int96Type>*>(row_group.next()), blob);
//    }
//    void operator()(row_group_writer& row_group, std::nullptr_t) const {
//       detail::write_one(static_cast<parquet::TypedColumnWriter<parquet::Int96Type>*>(row_group.next()), nullptr);
//    }
// };

template <typename Clock, typename Rep>
struct column_writer<std::chrono::time_point<Clock, std::chrono::duration<Rep, std::nano>>, typename std::is_integral<Rep>::type> {
   void operator()(row_group_writer& row_group, std::chrono::time_point<Clock, std::chrono::duration<Rep, std::nano>> const& value) const {
      auto blob = parquet::Int96{};
      // TODO:ditch internal function
      parquet::internal::NanosecondsToImpalaTimestamp(static_cast<std::int64_t>(value.time_since_epoch().count()), &blob);
      detail::write_one(static_cast<parquet::TypedColumnWriter<parquet::Int96Type>*>(row_group.next()), blob);
   }
   void operator()(row_group_writer& row_group, std::nullptr_t) const {
      detail::write_one(static_cast<parquet::TypedColumnWriter<parquet::Int96Type>*>(row_group.next()), nullptr);
   }
};

template <typename Clock, typename Rep>
struct column_writer<std::chrono::time_point<Clock, std::chrono::duration<Rep, std::micro>>, typename std::is_integral<Rep>::type> {
   void operator()(row_group_writer& row_group, std::chrono::time_point<Clock, std::chrono::duration<Rep, std::micro>> const& value) const {
      column_writer<Rep>{}(row_group, value.time_since_epoch().count());
   }
   void operator()(row_group_writer& row_group, std::nullptr_t) const {
      column_writer<Rep>{}(row_group, nullptr);
   }
};

template <typename Clock, typename Rep, typename Period>
struct column_writer<std::chrono::time_point<Clock, std::chrono::duration<Rep, Period>>, typename std::conjunction<std::is_integral<Rep>, std::ratio_greater_equal<Period, std::milli>>::type> {
   void operator()(row_group_writer& row_group, std::chrono::time_point<Clock, std::chrono::duration<Rep, Period>> const& value) const {
      auto dur = std::chrono::duration_cast<std::chrono::duration<std::uint64_t, std::milli>>(value.time_since_epoch());
      column_writer<Rep>{}(row_group, dur.count());
   }
   void operator()(row_group_writer& row_group, std::nullptr_t) const {
      column_writer<Rep>{}(row_group, nullptr);
   }
};

template <typename T>
struct column_writer<std::optional<T>, std::true_type> {
   void operator()(row_group_writer& row_group, std::optional<T> const& value) const {
      if (value.has_value())
         pqx::column_writer<T>{}(row_group, value.raw_value());
      else
         pqx::column_writer<T>{}(row_group, nullptr);
   }
};

template <typename... T>
class writer {
 public:
   writer(std::shared_ptr<arrow::io::OutputStream> sink,
          std::shared_ptr<parquet::WriterProperties> properties = parquet::default_writer_properties());

   template <typename... Options>
   writer(options<Options...>,
          std::shared_ptr<arrow::io::OutputStream> sink,
          std::shared_ptr<parquet::WriterProperties> properties = parquet::default_writer_properties());

   void push_back(T const&... value) {
      auto row_group = row_group_writer{writer_};
      (..., write(row_group, value));
   }

 private:
   template <typename U>
   void write(row_group_writer& row_group, U&& value);

 private:
   std::shared_ptr<parquet::schema::GroupNode> schema_;
   std::unique_ptr<parquet::ParquetFileWriter> file_writer_;
   parquet::RowGroupWriter* writer_;
};

}} // namespace pqx::v1

template <typename T>
void pqx::v1::detail::write_one(parquet::TypedColumnWriter<T>* writer, typename T::c_type value) {
   static const auto def_levels = std::int16_t{1};
   static const auto rep_levels = std::int16_t{0};
   writer->WriteBatch(1, &def_levels, &rep_levels, &value);
}

template <typename T>
void pqx::v1::detail::write_one(parquet::TypedColumnWriter<T>* writer, std::nullptr_t) {
   static const auto def_levels = std::int16_t{0};
   static const auto rep_levels = std::int16_t{0};
   writer->WriteBatch(1, &def_levels, &rep_levels, nullptr);
}

template <typename... T>
template <typename... Options>
pqx::v1::writer<T...>::writer(
    options<Options...>,
    std::shared_ptr<arrow::io::OutputStream> sink,
    std::shared_ptr<parquet::WriterProperties> properties /*= parquet::default_writer_properties()*/)
    : schema_{make_schema<boost::mp11::mp_list<T...>, Options...>()}, file_writer_{parquet::ParquetFileWriter::Open(std::move(sink), schema_, properties)}, writer_{file_writer_->AppendBufferedRowGroup()} {
}

template <typename... T>
pqx::v1::writer<T...>::writer(std::shared_ptr<arrow::io::OutputStream> sink,
                              std::shared_ptr<parquet::WriterProperties> properties /* = parquet::default_writer_properties()*/)
    : writer{options<>{}, sink, std::move(properties)} {
}

template <typename... T>
template <typename U>
void pqx::v1::writer<T...>::write(row_group_writer& row_group, U&& value) {
   using value_type = std::decay_t<U>;
   if constexpr (boost::describe::has_describe_members<value_type>::value) {
      using public_members = boost::describe::describe_members<value_type, boost::describe::mod_public>;
      boost::mp11::mp_for_each<public_members>([&row_group, &value, this](auto D) {
         // column = write(writer_, column, value.*D.pointer);
         write(row_group, value.*D.pointer);
         // auto column_writer = writer_->column(++column);
         // column = builder<typename decltype(D)::type>{}(writer, column, value.*D.pointer);
      });
   } else {
      column_writer<value_type>{}(row_group, value);
   }
}
