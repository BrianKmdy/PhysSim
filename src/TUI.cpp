#ifdef __unix__

#include "TUI.h"

#include <cmath>

TUI::TUI(float wWidth, float wHeight) : UserInterface(wWidth, wHeight)
{
/*    initscr();
    noecho();
    curs_set(0);
	nl();
    nodelay(stdscr, TRUE);

    getmaxyx(stdscr, height, width);

    win = newwin(height, width, 0, 0);
    */
}

void TUI::tick(const std::vector<Particle> * entities) {
//    werase(win);

    for (std::vector<Particle>::const_iterator it = entities->begin(); it != entities->end(); ++it)
        drawParticle(*it);

//    wnoutrefresh(win);
//    doupdate();
}

void TUI::drawParticle(Particle particle) {
    int x = round((particle.getX() / wWidth) * width);
    int y = round(height - ((particle.getY() / wHeight) * height) - 1);

//    mvwaddch(win, y, x, 'o');

    /*
    char buffer[32];

    mvwaddstr(win, 0, 0, "X: ");
    sprintf(buffer, "%f", particle.getX());
    waddstr(win, buffer);
    mvwaddstr(win, 1, 0, "Y: ");
    sprintf(buffer, "%f", particle.getY());
    waddstr(win, buffer);
    */
}

bool TUI::shouldClose() {
    return false;
}

void TUI::terminate() {
//    endwin();
}

bool TUI::returnPressed() {
    // TODO
}

#endif
