#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

struct container
{
    int value;
    int i;
    int j;
};

container max(struct container a, struct container b, struct container c)
{
    if ((a.value >= b.value) && (a.value >= c.value))
        return a;

    if ((b.value >= a.value) && (b.value >= c.value))
        return b;

    return c;
}

container min(struct container top,
              struct container right,
              struct container topRight)
{
    struct container m;

    if (top.value < right.value)
        m = right;
    else
        m = top;

    if (m.value < topRight.value)
        return m;
    else
        return topRight;
}

void showRoute(std::vector<std::vector<int>> board, int x, int y)
{
    for (auto i = 0; i < (int)board.size(); ++i)
    {
        for (auto j = 0; j < (int)board[i].size(); ++j)
        {
            if ((i == x) && (j == y))
                std::cout << " [" << board[i][j] <<"] ";
            else
                std::cout << "  " << board[i][j] <<"  ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

static int kingTravel(std::vector<std::vector<int>> board)
{
    const int rowSize = board.size();
    const int colSize = board[0].size();

    int score = 0;

    if (rowSize != colSize)
        throw "row and must be the same size";

    int j = 0;
    int i = colSize - 1;
    for ( ; i > 0; )
    {
        container top = {
            .value = board[i - 1][j],
            .i     = i - 1,
            .j     = j
        };

        container right = {
            .value = board[i][j + 1],
            .i     = i,
            .j     = j + 1
        };

        container topRight = {
            .value = board[i - 1][j + 1],
            .i     = i - 1,
            .j     = j + 1
        };

        container rv = min(top, right, topRight);

        i = rv.i;
        j = rv.j;
        score += rv.value;
#ifdef DEBUG
        std:: cout << "(y) col : " << rv.i << std::endl;
        std:: cout << "(x) row : " << rv.j << std::endl;
        std:: cout << "val     : " << rv.value << std::endl;
        std:: cout << "sсore   : " << score << std::endl << std::endl;
        showRoute(board, rv.i, rv.j);
#endif /* DEBUG **/
    }

    if (j != rowSize)
    {
        score += board[i][j + 1];
    }

    return score;
}

static void parseArgs(char const *fileName, std::vector<std::vector<int>> &board)
{
    std::ifstream inFile(fileName);
    if (!inFile.is_open())
    {
        std::cerr << "Can't open input file" << std::endl;
        exit(0);
    }

    for (auto i = 0; i < 8; ++i) {
        std::string line;

        getline(inFile, line);

        std::stringstream stream(line);

        int value;

        std::vector<int> newRow;

        while (stream >> value)
        {
            newRow.push_back(value);
        }

        board.push_back(newRow);
    }

#ifdef DEBUG
    for_each(board.begin(), board.end(), [](std::vector<int> v)
    {
        for_each(v.begin(), v.end(), [](int v)
        {
            std::cout << v << " ";
        });
        std::cout << std::endl;
    });
#endif /* DEBUG */

    inFile.close();
}

static void parseArgsRunIt(char const *fileName)
{
    std::vector<std::vector<int>> board;

    parseArgs(fileName, board);

    std::cout << kingTravel(board) << std::endl;

    std::ofstream outFile("king2.out");
    
    if (!outFile.is_open())
        throw "Can't open input file";

    outFile << kingTravel(board);

    outFile.close();
}

static void usage(char const *app)
{
    std::cout << "usage: " << app << " <file name>" << std::endl;
}

int main(int argc, char const *argv[])
{
    try
    {
        if (argc != 1)
        {
            parseArgsRunIt(argv[1]);
        }
        else
        {
            //usage(argv[0]);
            parseArgsRunIt("king2.in");
        }
    }
    catch (std::string err)
    {
        std::cerr << err << std::endl;
        return -1;
    }

    return 0;
}
