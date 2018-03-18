#ifndef _GLOBAL_VARIABLES_H
#define _GLOBAL_VARIABLES_H

#include <mutex>

enum {
    GAME_START = 0,
    GAME_STOP  = 1,
    GAME_END   = 2 
};

class GlobalVariables
{
    private:
        int gameState_;
        int timeToRestart_;
        int timeToThink_;

        std::mutex mutex_;
    public:
        GlobalVariables();
        ~GlobalVariables();

        void setGameState(int newState);
        void setTimeToRestart(int newTime);
        void setTimeToThink(int newTime);

        int getGameState();
        int getTimeToRestart();
        int getTimeToThink();
};

#endif /* _GLOBAL_VARIABLES_H */
