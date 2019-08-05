#ifdef __unix__

#ifndef TUI_H
#define TUI_H

//#include <ncurses.h>
#include <vector>
#include <cstdlib>

#include "Particle.h"
#include "UserInterface.h"

class TUI : public UserInterface
{
    private:
//        WINDOW *win;

    public:
        TUI(float wWidth, float wHeight);

        void tick(const std::vector<Particle> * entities);
        void drawParticle(Particle particle);
        bool shouldClose();
        void terminate();
        bool returnPressed();
};

#endif // TUI_H

#endif
