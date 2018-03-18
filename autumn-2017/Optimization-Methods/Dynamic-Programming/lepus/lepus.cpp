#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>

std::pair<int, int> max(int a, int b)
{
    if (a > b)
        return std::make_pair(1, a);
    else
        return std::make_pair(3, b);
}

std::pair<int, int> max(int a, int b, int c)
{
    if ((a > b) && (a > c))
        return std::make_pair(1, a);

    if ((b > a) && (b > c))
        return std::make_pair(3, b);

    return std::make_pair(5, c);
}

static int lepusTravel(std::vector<int> &trail,
                       std::vector<int> &score,
                       std::vector<int> &path)
{
    auto size = static_cast<int>(trail.size());
    auto scorePoints = 0;
    auto i = 0;

    //std::cout << "# Trail size = " << trail.size() << std::endl;

    while (size)
    {
        if ((size - 5) > 1)
        {
            //std::cout << "** Size - 5" << std::endl;
            
            // Get max score point between: +1, +3, +5, step
            auto tmp = max(trail[i + 1], trail[i + 3], trail[i + 5]);
            if (tmp.second == -1)
            {
                return -1;
            }
            else
            {
                score.push_back(tmp.second);
            }
            
            //std::cout << "** Size - 5 value  = " << tmp.second << std::endl;
            //std::cout << "** Size - 5 index += " << tmp.first << std::endl;
            
            size -= tmp.first;
            i += tmp.first;
        }
        else if ((size - 3) > 1)
        {
            //std::cout << "** Size - 3" << std::endl;
            
            // Get max score point between: +1, +3
            auto tmp = max(trail[i + 1], trail[i + 3]);
            if (tmp.second == -1)
            {
                return -1;
            }
            else
            {
                score.push_back(tmp.second);
            }

            //std::cout << "** Size - 3 value  = " << tmp.second << std::endl;
            //std::cout << "** Size - 3 index += " << tmp.first << std::endl;
            
            size -= tmp.first;
            i += tmp.first;
        }
        else if ((size - 1) > 0)
        {
            //std::cout << "** Size - 1" << std::endl;
            
            // Get max score
            if (trail[i + 1] == -1)
            {
                return -1;
            }
            else
            {
                score.push_back(trail[i + 1]);
            }

            //std::cout << "** Size - 1 value  = " << trail[i + 1] << std::endl;
            //std::cout << "** Size - 1 index += " << 1 << std::endl;

            size -= 1;
            i += 1;
        }
        else
        {
            //std::cout << "** Size - 0" << std::endl;
            
            // Get max score
            if (trail[i] == -1)
            {
                return -1;
            }
            else
            {
                score.push_back(trail[i]);
            }

            //std::cout << "** Size - 1 value  = " << trail[i] << std::endl;
            //std::cout << "** Size - 1 index += " << 1 << std::endl;

            size -= 1;
            i += 1;
        }

        //std::cout << "size : " << size << std::endl;
        //std::cout << "i    : " << i << std::endl;
    }

    std::for_each(score.begin(), score.end(), [&](int &value)
    {
        scorePoints += value; 
    });

    return scorePoints;
}

/*****************************************************************************/
/* w - swamp cell   (-1 -> exit)                                             */
/* " - grass cell   (+1)                                                     */
/* . - regular cell (0)                                                      */
/*****************************************************************************/
static void parseArgs(char const *fileName, std::vector<int> &v, int &size)
{
    char ch;
    std::string line;
    std::ifstream inFile(fileName);
    
    if (!inFile.is_open())
    {
        throw "Can't open input file";
    }
        
    getline(inFile, line); // Read first line

    size = std::stoi(line);
    if (size < 2 || size > 1000)
    {
        throw "Not valid cell number";
    }
    
    getline(inFile, line); // Read second line

    std::stringstream stream(line);

    while (stream >> ch)
    {
        if (ch == 'w')
        {
            v.push_back(-1);
        }
        else if (ch == '"')
        {
            v.push_back(1);
        }
        else if (ch == '.')
        {
            v.push_back(0);
        }
    }

    inFile.close();
}

static void initData(std::vector<int> &v, int &size)
{
    size = 4;
    std::string data = ".\"\".";
    std::stringstream stream(data);
    char ch;

    while (stream >> ch)
    {
        if (ch == 'w')
        {
            v.push_back(-1);
        }
        else if (ch == '"')
        {
            v.push_back(1);
        }
        else if (ch == '.')
        {
            v.push_back(0);
        }
    }
}

static void parseArgsRunIt(char const *fileName)
{
    auto size = 0;
    std::vector<int> trail;
    std::vector<int> score;
    std::vector<int> path;

    if (fileName)
    {
        parseArgs(fileName, trail, size);
    }
    else
    {
        parseArgs("lepus.in", trail, size);
        //initData(trail, size);
    }

    auto answer = lepusTravel(trail, score, path);

    std::cout << "Lepus travel score : " << answer << std::endl;

    std::ofstream outFile("lepus.out");
    
    if (!outFile.is_open())
        throw "Can't open input file";

    outFile << answer;

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
            parseArgsRunIt(NULL);
        }
    } 
    catch (std::string err)
    {
        std::cerr << err << std::endl;
        return -1;
    }

    return 0;
}
