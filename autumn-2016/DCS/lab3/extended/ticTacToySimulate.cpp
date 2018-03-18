#include "ticTacToySimulate.h"

#include <iostream>
#include <thread>

#include <random>
#include <chrono>

using namespace std;

TicTacToySimulate::TicTacToySimulate(GlobalVariables *parametrs)
{
    this->parametrs_ = parametrs;

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            gameFiled_[i][j] = '.';
        }
    }
}

TicTacToySimulate::~TicTacToySimulate() {}

void TicTacToySimulate::printGameFiled()
{
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            cout << gameFiled_[i][j] << " ";
        }
        cout << endl;
    }
}

void TicTacToySimulate::makeMove(char player)
{
    auto seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0, 2);

    int x = distribution(generator);
    int y = distribution(generator);

    while (1) {
        int x = distribution(generator);
        int y = distribution(generator);

        if (gameFiled_[x][y] == '.') {
            gameFiled_[x][y] = player;
            printGameFiled();
            return;
        } else {
            for (auto i = 0; i < 3; ++i) {
                for (auto j = 0; j < 3; ++j) {
                    if (gameFiled_[i][j] == '.') {
                        gameFiled_[i][j] = player;
                        printGameFiled();
                        return;
                    }
                }
            }
        }
    }
}

int TicTacToySimulate::checkWin()
{
    // Check by rows
    if ( (gameFiled_[0][0] == 'x') &&
         (gameFiled_[0][1] == 'x') &&
         (gameFiled_[0][2] == 'x') )
    {
        cout << "1 X won!" << endl;
        return 1;
    }

    if ( (gameFiled_[1][0] == 'x') &&
         (gameFiled_[1][1] == 'x') &&
         (gameFiled_[1][2] == 'x') )
    {
        cout << "2 X won!" << endl;
        return 1;
    }

    if ( (gameFiled_[2][0] == 'x') &&
         (gameFiled_[2][1] == 'x') &&
         (gameFiled_[2][2] == 'x') )
    {
        cout << "3 X won!" << endl;
        return 1;
    }

    // Check by collumn
    if ( (gameFiled_[0][0] == 'x') &&
         (gameFiled_[1][0] == 'x') &&
         (gameFiled_[2][0] == 'x') )
    {
        cout << "4 X won!" << endl;
        return 1;
    }

    if ( (gameFiled_[0][1] == 'x') &&
         (gameFiled_[1][1] == 'x') &&
         (gameFiled_[2][1] == 'x') )
    {
        cout << "5 X won!" << endl;
        return 1;
    }

    if ( (gameFiled_[0][2] == 'x') &&
         (gameFiled_[1][2] == 'x') &&
         (gameFiled_[2][2] == 'x') )
    {
        cout << "6 X won!" << endl;
        return 1;
    }


    // Check by rows
    if ( (gameFiled_[0][0] == 'o') &&
         (gameFiled_[0][1] == 'o') &&
         (gameFiled_[0][2] == 'o') )
    {
        cout << "7 O won!" << endl;
        return 1;
    }

    if ( (gameFiled_[1][0] == 'o') &&
         (gameFiled_[1][1] == 'o') &&
         (gameFiled_[1][2] == 'o') )
    {
        cout << "8 O won!" << endl;
        return 1;
    }

    if ( (gameFiled_[2][0] == 'o') &&
         (gameFiled_[2][1] == 'o') &&
         (gameFiled_[2][2] == 'o') )
    {
        cout << "9 O won!" << endl;
        return 1;
    }


    // Check by collumn O
    if ( (gameFiled_[0][0] == 'o') &&
         (gameFiled_[1][0] == 'o') &&
         (gameFiled_[2][0] == 'o') )
    {
        cout << "10 O won!" << endl;
        return 1;
    }

    if ( (gameFiled_[0][1] == 'o') &&
         (gameFiled_[1][1] == 'o') &&
         (gameFiled_[2][1] == 'o') )
    {
        cout << "11 O won!" << endl;
        return 1;
    }

    if ( (gameFiled_[0][2] == 'o') &&
         (gameFiled_[1][2] == 'o') &&
         (gameFiled_[2][2] == 'o') )
    {
        cout << "12 O won!" << endl;
        return 1;
    }

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            if (gameFiled_[i][j] == '.') {
                return 0;
            }
        }
    }

    cout << "No one won!" << endl;
    return 1;
}

void TicTacToySimulate::runGame_(int think, int restart)
{
    cout << "Game started think " << think << " restart " << restart << endl;
    for (auto i = 0; i < 9; ++i) {
        this_thread::sleep_for(chrono::seconds(think));
        
        if (this->parametrs_->getGameState() == GAME_START) {
            cout << "Player 1 - make move" << endl;
            makeMove('x');
            if (checkWin() == 1)
                break;

            cout << "Player 2 - make move" << endl;
            makeMove('o');
            if (checkWin() == 1)
                break;

        } else if (this->parametrs_->getGameState() == GAME_END) {
            return;
        } else {
            --i;
        }
    }

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            gameFiled_[i][j] = '.';
        }
    }

    this_thread::sleep_for(chrono::seconds(restart));
}

void TicTacToySimulate::runGame()
{
    while (this->parametrs_->getGameState() != GAME_END) {
        runGame_(this->parametrs_->getTimeToThink(), this->parametrs_->getTimeToRestart());
    }
}

void createAndRunKeyGame(GlobalVariables *parametrs)
{
    TicTacToySimulate ticTacToySimulate(parametrs);

    ticTacToySimulate.runGame();
}