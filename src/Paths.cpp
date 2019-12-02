#include "Paths.h"

const std::string ConfigFileName = "config.yaml";
const std::string PositionDirectoryName = "position";

const std::filesystem::path ConfigFilePath = std::filesystem::path(ConfigFileName);

const std::filesystem::path OutputDirectory = std::filesystem::path("latest");
const std::filesystem::path PositionDataDirectory = OutputDirectory / PositionDirectoryName;
const std::filesystem::path StateDataDirectory = OutputDirectory / std::filesystem::path("state");

const std::filesystem::path OutputConfigFilePath = OutputDirectory / std::filesystem::path(ConfigFileName);
const std::filesystem::path LogFilePath = OutputDirectory / std::filesystem::path("log.txt");