#include "keyBoardReader.h"

#include <iostream>

using namespace std;

KeyBoardReader::KeyBoardReader(GlobalVariables *parametrs)
{
    this->parametrs_ = parametrs;
}

KeyBoardReader::~KeyBoardReader() {}

int KeyBoardReader::checkGame()
{
    if (parametrs_->getGameState() == GAME_STOP) {
        cout << "GAME_START" << endl;
        return GAME_START;
    }
    
    if (parametrs_->getGameState() == GAME_START) {
        cout << "GAME_STOP" << endl;
        return GAME_STOP;
    }  
}

int KeyBoardReader::readTime(int lb, int ub)
{
    int time;

    while (1) {
        cout << "Input time (" << lb << " - " << ub << ")" << endl;
        cin >> time;

        if ((time > lb) && (time < ub)) {
            break;
        }
    }

    return time;
}

void KeyBoardReader::readKey(char c)
{
    switch (c)
    {
        case 's':
            parametrs_->setGameState(checkGame());
            break;

        case 'q':
            parametrs_->setGameState(GAME_END);
            break;

        case 'a':
            parametrs_->setTimeToRestart(readTime(1, 10));
            break;

        case 't':
            parametrs_->setTimeToThink(readTime(5, 30));
            break;

        default:
            break;
    }
}


void createAndRunKeyReader(GlobalVariables *parametrs)
{
    KeyBoardReader keyBoardReader(parametrs);

    while (parametrs->getGameState() != GAME_END) {
        char c = getchar();

        switch (c) 
        {
            case 's':
                cout << "Changing game state: ";
                keyBoardReader.readKey('s');
                break;

            case 'q':
                keyBoardReader.readKey('q');
                break;

            case 'a':
                cout << "Changing time to restart" << endl;
                keyBoardReader.readKey('a');
                break;

            case 't':
                cout << "Changing time to think" << endl;
                keyBoardReader.readKey('t');
                break;

            default:
                break;
        }
    }

    cout << "Game close..." << endl;
}