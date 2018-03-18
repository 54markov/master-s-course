/*
 * https://rosettacode.org/wiki/Transportation_problem
 * https://www.slideshare.net/agniere68/transportation-problem-11656517
 * http://orms.pef.czu.cz/text/transProblem.html
 */
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <limits>

class Shipment
{
    public:
        double costPerUnit, quantity;
        int r, c;

        Shipment(double _q = -1.0, double _cpu = -1.0, int _r = -1, int _c = -1)
        {
            this->quantity    = _q;
            this->costPerUnit = _cpu;
            this->r           = _r;
            this->c           = _c;
        }

        bool isNull()
        {
            if ((this->quantity    == -1.0) &&
                (this->costPerUnit == -1.0) &&
                (this->r           == -1)   &&
                (this->c           == -1))
            {
                return true;
            }
            return false;
        }

        bool operator!=(const Shipment &rhs)
        {
            if ((this->quantity    == rhs.quantity)    &&
                (this->costPerUnit == rhs.costPerUnit) &&
                (this->r           == rhs.r)           &&
                (this->c           == rhs.c))
            {
                return false;
            }
            return true;
        }

        void dump()
        {
            std::cout << "[" << this->quantity << ", "
                      << this->costPerUnit << ","
                      << this->r << ", "
                      << this->c << "]\t";
        }
};

class TransportationProblem
{
    private:
        std::vector<int> demand;
        std::vector<int> supply;
        std::vector<std::vector<double>> costs;
        std::vector<std::vector<Shipment>> matrix;

        std::vector<Shipment> matrixToVector();
        std::vector<Shipment> getNeighbors(Shipment s, std::vector<Shipment> lst);
        std::vector<Shipment> getClosedPath(Shipment s);
        void fixDegenerateCase();

    public:
        TransportationProblem();
        ~TransportationProblem();

        void init(std::string filename);
        void northWestCornerRule();
        void steppingStone();
        void printResult(std::string filename);
    
};

TransportationProblem::TransportationProblem() { }
TransportationProblem::~TransportationProblem() { }

void TransportationProblem::init(std::string filename)
{
    std::ifstream file(filename);

    try
    {
        if (!file.is_open())
            throw std::string("Can't read from file");

        int numSources      = 0;
        int numDestinations = 0;

        int totalSrc        = 0;
        int totalDst        = 0;
        std::string line;
        std::stringstream ss;
        std::stringstream ss1;
        std::stringstream ss2;

        getline(file, line);
        ss << line;
        ss >> numSources;
        ss >> numDestinations;

        std::vector<int> src;
        std::vector<int> dst;

        getline(file, line);
        ss1 << line;
        for (auto i = 0; i < numSources; i++)
        {
            int tmp = 0;
            ss1 >> tmp;
            src.push_back(tmp);
            totalSrc += tmp;
        }

        getline(file, line);
        ss2 << line; 
        for (auto i = 0; i < numDestinations; i++)
        {
            int tmp = 0;
            ss2 >> tmp;
            dst.push_back(tmp);
            totalDst += tmp;
        }

        // Fix imbalance
        if (totalSrc > totalDst)
        {
            dst.push_back(totalSrc - totalDst);
        }
        else if (totalDst > totalSrc)
        {
            src.push_back(totalDst - totalSrc);
        }

        this->supply = src;
        this->demand = dst;

        for (auto i = 0; i < static_cast<int>(this->supply.size()); i++)
        {
            std::vector<double> v(this->demand.size(), 0.0);
            this->costs.push_back(v);
        }

        for (auto i = 0; i < static_cast<int>(supply.size()); i++)
        {
            std::vector<Shipment> v(this->demand.size(), Shipment());
            this->matrix.push_back(v);
        }
 
        for (auto i = 0; i < numSources; i++)
        {
            std::stringstream ss;
            getline(file, line);
            ss << line;
            for (auto j = 0; j < numDestinations; j++)
            {
                double tmp = 0.0;
                ss >> tmp;
                this->costs[i][j] = (double)tmp;
            }
        }

        file.close();
    }
    catch (std::string err)
    {
        std::cerr << err << std::endl;
        return;
    }
}

void TransportationProblem::printResult(const std::string filename)
{
    std::cout << "Optimal solution " << filename << std::endl;

    double totalCosts = 0.0;

    for (auto r = 0; r < static_cast<int>(this->supply.size()); r++)
    {
        for (auto c = 0; c < static_cast<int>(this->demand.size()); c++)
        { 
            Shipment shipment = this->matrix[r][c];
            if (!shipment.isNull() && shipment.r == r && shipment.c == c)
            {
                std::cout << "\t" << shipment.quantity << "\t";
                totalCosts += (shipment.quantity * shipment.costPerUnit);
            }
            else
            {
                std::cout << "\t-\t";
            }
        }
        std::cout << std::endl;
    }
    std::cout << "Total costs: " << totalCosts << std::endl;
}

void TransportationProblem::northWestCornerRule()
{
    for (auto r = 0, northwest = 0; r < static_cast<int>(this->supply.size()); r++)
    {
        for (auto c = northwest; c < static_cast<int>(this->demand.size()); c++)
        {
            auto quantity = static_cast<double>(std::min(this->supply[r], this->demand[c]));
            if (quantity > 0.0)
            {
                this->matrix[r][c]  = Shipment((double)quantity, this->costs[r][c], r, c);
                this->supply[r]    -= quantity;
                this->demand[c]    -= quantity;
 
                if (supply[r] == 0)
                {
                    northwest = c;
                    break;
                }
            }
        }
    }
}

void TransportationProblem::steppingStone()
{
    double maxReduction = 0.0;

    std::vector<Shipment> move;

    Shipment leaving;

    this->fixDegenerateCase();

    for (int r = 0; r < static_cast<int>(supply.size()); r++)
    {
        for (int c = 0; c < static_cast<int>(demand.size()); c++)
        {
            if (!matrix[r][c].isNull())
            {
                continue;
            }

            Shipment trial(0.0, costs[r][c], r, c);
            std::vector<Shipment> path = getClosedPath(trial);

            double reduction = 0.0;
            double lowestQuantity = std::numeric_limits<double>::max();
            
            Shipment leavingCandidate;

            auto plus = true;
            for (auto s : path)
            {
                if (plus)
                {
                    reduction += s.costPerUnit;
                }
                else
                {
                    reduction -= s.costPerUnit;
                    if (s.quantity < lowestQuantity)
                    {
                        leavingCandidate = s;
                        lowestQuantity   = s.quantity;
                    }
                }
                plus = !plus;
            }

            if (reduction < maxReduction)
            {
                move         = path;
                leaving      = leavingCandidate;
                maxReduction = reduction;
            }
        }
    }

    if (move.size() != 0)
    {
        double q  = leaving.quantity;
        bool plus = true;

        for (Shipment s : move)
        {
            Shipment nullObj;
            s.quantity += plus ? q : -q;
            this->matrix[s.r][s.c] = s.quantity == 0 ? nullObj : s;
            plus = !plus;
        }
        this->steppingStone();
    }
}

std::vector<Shipment> TransportationProblem::matrixToVector()
{
    std::vector<Shipment> v;

    for (auto row : this->matrix)
    {
        for (auto s: row)
        {
            if (!s.isNull())
            {
                v.push_back(s);
            }
        }
    }
    return v;
}

std::vector<Shipment> TransportationProblem::getNeighbors(Shipment s, std::vector<Shipment> lst)
{
    std::vector<Shipment> nbrs(2);
    
    for (Shipment o : lst)
    {
        if (o != s)
        {
            if (o.r == s.r && nbrs[0].isNull())
            {
                nbrs[0] = o;
            }
            else if (o.c == s.c && nbrs[1].isNull())
            {
                nbrs[1] = o;
            }
            if (!nbrs[0].isNull() && !nbrs[1].isNull())
            {
                break;
            }
        }
    }

    return nbrs;
}

std::vector<Shipment> TransportationProblem::getClosedPath(Shipment s)
{
    std::vector<Shipment> path;
    
    path.push_back(s);

    for (auto i : this->matrixToVector())
    {
        path.push_back(i);
    }

    /*
     * Remove (and keep removing) elements that do not have a
     * vertical AND horizontal neighbor
     */
    while (true)
    {
        bool isDone = true;
        std::vector<Shipment> tmpPath;
        for (auto iter : path)
        {
            std::vector<Shipment> nbrs = getNeighbors(iter, path);
            if (nbrs[0].isNull() || nbrs[1].isNull())
            {
                isDone = false;
                continue;
            }
            else
            {
                tmpPath.push_back(iter);
            }
        }
        path = tmpPath;
        
        if (isDone)
        {
            break;
        }
    }

    // Place the remaining elements in the correct plus-minus order
    std::vector<Shipment> stones = path;
    Shipment prev = s;

    for (auto i = 0; i < static_cast<int>(stones.size()); i++)
    {
        stones[i] = prev;
        prev = getNeighbors(prev, path)[i % 2];
    }
    return stones;
}

void TransportationProblem::fixDegenerateCase()
{
    double eps = std::numeric_limits<double>::min();

    if (this->supply.size() + this->demand.size() - 1 != this->matrixToVector().size())
    {
        for (auto r = 0; r < static_cast<int>(this->supply.size()); r++)
        {
            for (auto c = 0; c < static_cast<int>(this->demand.size()); c++)
            {
                if (matrix[r][c].isNull())
                {
                    Shipment dummy(eps, this->costs[r][c], r, c);
                    if (getClosedPath(dummy).size() == 0)
                    {
                        this->matrix[r][c] = dummy;
                        return;
                    }
                }
            }
        }
    }
}

int main(int argc, char const *argv[])
{
    std::vector<std::string> files;

    files.push_back("input1.txt");
    files.push_back("input2.txt");
    files.push_back("input3.txt");

    std::for_each(files.begin(), files.end(), [](std::string filename)
    {
        TransportationProblem p1;
        p1.init(filename);
        p1.northWestCornerRule();
        p1.steppingStone();
        p1.printResult(filename);
    });

    return 0;
}