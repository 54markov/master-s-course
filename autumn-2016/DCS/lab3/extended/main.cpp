#include <iostream>
#include <thread>

#include "globalVariables.h"
#include "keyBoardReader.h"
#include "ticTacToySimulate.h"

using namespace std;

int main(int argc, char const *argv[])
{
    GlobalVariables parametrs;

    thread key  (createAndRunKeyReader, &parametrs);
    thread game (createAndRunKeyGame, &parametrs);

    key.join();
    game.join();

    return 0;
}