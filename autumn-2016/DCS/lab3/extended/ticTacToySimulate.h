#ifndef _TIC_TAC_TOY_SIMULATE_H
#define _TIC_TAC_TOY_SIMULATE_H

#include "globalVariables.h"

class TicTacToySimulate
{
    private:
        char gameFiled_[3][3];
        GlobalVariables *parametrs_;
    public:
        TicTacToySimulate(GlobalVariables *parametrs);
        ~TicTacToySimulate();

        void runGame();
        void runGame_(int think, int restart);

        void printGameFiled();
        void makeMove(char player);
        int checkWin();
};

void createAndRunKeyGame(GlobalVariables *parametrs);

#endif /* _TIC_TAC_TOY_SIMULATE_H */
