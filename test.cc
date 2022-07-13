#include <iostream>
#include <yaml-cpp/yaml.h>

int main() {
    std::cout << "Cmake success!" << std::endl;

    for(auto i : {1,2,3}) {
        std::cout << i;
    }
    std::cout << std::endl;

    std::cout << "The docker build is working!!!!" << std::endl;
}