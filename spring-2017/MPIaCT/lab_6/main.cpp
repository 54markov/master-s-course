#include <iostream>
#include <cmath>

#include <random>

#define exp 2.71828182846

using namespace std;

class QueuingSystem
{
    private:
        double lambda_;
        double mu_;

        int requestCounter_;
        int simulatedTicks_;

        int statRequestCounter_;
        int statHandlingCounter_;

    public:
        /* Constructor */
        QueuingSystem(double lambda, double mu, int ticks);
        /* Destructor */
        ~QueuingSystem() {}

        /* Prototypes */
        double calcIncomingProbability(int elapsedTicksFromLastRequest);
        double calcHandlingProbability(int elapsedTicksFromLastHandling);
        void runSimulate();
        void printStat(int ticks, bool isBusy);
};

/* Implementation */
QueuingSystem::QueuingSystem(double lambda, double mu, int ticks)
{
    lambda_              = lambda;
    mu_                  = mu;
    requestCounter_      = 0;
    simulatedTicks_      = ticks;
    statRequestCounter_  = 0;
    statHandlingCounter_ = 0;
}

double QueuingSystem::calcIncomingProbability(int elapsedTicksFromLastRequest)
{
    double res = 1.0 - pow(exp, -(this->lambda_ * (double)elapsedTicksFromLastRequest));

    return res;
}

double QueuingSystem::calcHandlingProbability(int elapsedTicksFromLastHandling)
{
    double res = 1.0 - pow(exp, -(this->mu_ * (double)elapsedTicksFromLastHandling));

    return res;
}

void QueuingSystem::printStat(int ticks, bool isBusy)
{
    cout << (isBusy ? "busy" : "free");
    cout << "\t" << this->lambda_;
    cout << "\t" << this->mu_;
    cout << "\t" << ticks;
    cout << "\t" << this->requestCounter_ << endl;

    if (ticks == this->simulatedTicks_) {
        cout << endl;
        cout << "Statisic of requests  = " << this->statRequestCounter_ << endl;
        cout << "Statisic of handling  = " << this->statHandlingCounter_ << endl;
    }
}

void QueuingSystem::runSimulate()
{
    int elapsedTicksFromLastRequest  = 0;
    int elapsedTicksFromLastHandling = 0;
    bool isBusy = false;

    cout << "\nSS  - System state"<< endl;
    cout << "ET  - Elapsed ticks"<< endl;
    cout << "AoR - Amount of requests"<< endl;
    cout << "\nSS \tλ \tμ \tET \tAoR"<< endl;

    for (auto i = 0; i < this->simulatedTicks_; ++i) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        elapsedTicksFromLastRequest++;
        elapsedTicksFromLastHandling++;

        double p_lambda = this->calcIncomingProbability(elapsedTicksFromLastRequest);
        double rand = dis(gen);
        if (rand < p_lambda) {
            isBusy = true;
            this->requestCounter_++;
            this->statRequestCounter_++;
            elapsedTicksFromLastRequest = 0;
        }

        if (this->requestCounter_ > 0) {
            double p_mu = this->calcHandlingProbability(elapsedTicksFromLastHandling);
            rand = dis(gen);
            if (rand < p_mu) {
                isBusy = false;
                this->requestCounter_--;
                this->statHandlingCounter_++;
                elapsedTicksFromLastHandling = 0;
            }
        }

        if (i % 100 == 0) {
            this->printStat(i, isBusy);
        }
    }


    this->printStat(this->simulatedTicks_, false);
    cout << "*********************************************" << endl;
}

int main(int argc, char const *argv[])
{
    QueuingSystem qs0(1.1, 0.97, 1000);
    QueuingSystem qs1(1.1, 1.0, 1000);
    QueuingSystem qs2(1.1, 1.1, 1000);
    QueuingSystem qs3(1.1, 1.5, 1000);

    qs0.runSimulate();
    qs1.runSimulate();
    qs2.runSimulate();
    qs3.runSimulate();

    return 0;
}
