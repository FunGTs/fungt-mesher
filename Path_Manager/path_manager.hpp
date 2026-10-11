#if !defined(_PATH_MANAGER_H_)
#define _PATH_MANAGER_H_
#include <filesystem>
#include <stdexcept>
#include <string>
std::filesystem::path findProjectRoot();
std::string getAssetPath(const std::string &path);
#endif // _PATH_MANAGER_H_
