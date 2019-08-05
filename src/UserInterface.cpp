#include "UserInterface.h"

UserInterface::UserInterface(float wWidth, float wHeight, int width, int height)
{
    this->wWidth = wWidth;
    this->wHeight = wHeight;
    this->width = width;
    this->height = height;
    selectedEntityIndex = 0;
}
