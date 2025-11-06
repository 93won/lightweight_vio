/**
 * @file      StringUtils.h
 * @brief     Common string utility functions for lightweight VIO.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-06
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef LIGHTWEIGHT_VIO_STRING_UTILS_H
#define LIGHTWEIGHT_VIO_STRING_UTILS_H

#include <string>
#include <vector>

namespace lightweight_vio {
namespace utils {

/**
 * @brief Trim whitespace characters from both ends of a string
 * @param str Input string
 * @return Trimmed string
 */
std::string trim(const std::string& str);

/**
 * @brief Split a string by delimiter
 * @param str Input string
 * @param delimiter Delimiter character
 * @return Vector of split strings
 */
std::vector<std::string> split(const std::string& str, char delimiter);

} // namespace utils
} // namespace lightweight_vio

#endif // LIGHTWEIGHT_VIO_STRING_UTILS_H
