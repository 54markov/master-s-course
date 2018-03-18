#include "globalVariables.h"

using namespace std;

GlobalVariables::GlobalVariables()
{
    gameState_     = GAME_STOP;
    timeToThink_   = 5;
    timeToRestart_ = 1;
}

GlobalVariables::~GlobalVariables() { }

void GlobalVariables::setGameState(int newState)
{
    this->mutex_.lock();
    gameState_ = newState;
    this->mutex_.unlock();
}

void GlobalVariables::setTimeToRestart(int newTime)
{
    this->mutex_.lock();
    timeToRestart_ = newTime;
    this->mutex_.unlock();
}

void GlobalVariables::setTimeToThink(int newTime)
{
    this->mutex_.lock();
    timeToThink_ = newTime;
    this->mutex_.unlock();
}

int GlobalVariables::getGameState()
{
    int state;

    this->mutex_.lock();
    state = gameState_;
    this->mutex_.unlock();

    return state;
}

int GlobalVariables::getTimeToRestart()
{
    int time;

    this->mutex_.lock();
    time = timeToRestart_;
    this->mutex_.unlock();

    return time;
}

int GlobalVariables::getTimeToThink()
{
    int time;

    this->mutex_.lock();
    time = timeToThink_;
    this->mutex_.unlock();

    return time;
}
