#include "Paths.h"

const std::filesystem::path ConfigFilePath = std::filesystem::path("config.yaml");

const std::filesystem::path OutputDirectory = std::filesystem::path("latest");
const std::filesystem::path PositionDataDirectory = OutputDirectory / std::filesystem::path("position");
const std::filesystem::path StateDataDirectory = OutputDirectory / std::filesystem::path("state");

const std::filesystem::path OutputConfigFilePath = OutputDirectory / std::filesystem::path("config.yaml");
const std::filesystem::path LogFilePath = OutputDirectory / std::filesystem::path("log.txt");