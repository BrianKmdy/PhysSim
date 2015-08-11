#include "UserInterface.h"

UserInterface::UserInterface(double wWidth, double wHeight, int width, int height)
{
    this->wWidth = wWidth;
    this->wHeight = wHeight;
    this->width = width;
    this->height = height;
    selectedEntityIndex = 0;
}
