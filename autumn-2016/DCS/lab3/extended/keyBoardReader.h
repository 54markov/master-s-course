#ifndef _KEY_BOARD_READER_H
#define _KEY_BOARD_READER_H

#include "globalVariables.h"

class KeyBoardReader
{
    private:
        GlobalVariables *parametrs_;

    public:
        KeyBoardReader(GlobalVariables *parametrs);
        ~KeyBoardReader();

        void readKey(char c);
        int checkGame();
        int readTime(int lb, int ub);
    
};

void createAndRunKeyReader(GlobalVariables *parametrs);

#endif /* _KEY_BOARD_READER_H */
