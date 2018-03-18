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

        int simulatedTicks_;

        int handlingCounter_;
        int rejectingCounter_;
        int idleCounter_;

        double  c1_;
        double  c2_;
        double  c3_;
        double  c4_;

    public:
        /* Constructor */
        QueuingSystem(double lambda,
                      double mu,
                      int ticks,
                      double c1,
                      double c2,
                      double c3,
                      double c4);
        /* Destructor */
        ~QueuingSystem() {}

        /* Prototypes */
        double calcIncomingProbability(int elapsedTicksFromLastRequest);
        double calcHandlingProbability(int elapsedTicksFromLastHandling);
        void runSimulate();
        void printStat(int ticks, bool isBusy);
};

/* Implementation */
QueuingSystem::QueuingSystem(double lambda,
                             double mu,
                             int ticks,
                             double c1,
                             double c2,
                             double c3,
                             double c4)
{
    lambda_           = lambda;
    mu_               = mu;
    simulatedTicks_   = ticks;
    handlingCounter_  = 0;
    rejectingCounter_ = 0;
    idleCounter_      = 0;
    c1_               = c1;
    c2_               = c2;
    c3_               = c3;
    c4_               = c4;
}

double QueuingSystem::calcIncomingProbability(int elapsedTicksFromLastRequest)
{
    double res = 1.0 - pow(exp, -(this->lambda_ * (double)elapsedTicksFromLastRequest));
    return res;
/*
    if (res == 1.0) {
        return 1;
    } else {
        return 0;
    }
*/
}

double QueuingSystem::calcHandlingProbability(int elapsedTicksFromLastHandling)
{
    double res = 1.0 - pow(exp, -(this->mu_ * (double)elapsedTicksFromLastHandling));
    return res;
/*
    if (res == 1.0) {
        return 1;
    } else {
        return 0;
    }
*/
}

void QueuingSystem::printStat(int ticks, bool isBusy)
{
    //cout << "\nSS \tλ \tμ \tc1 \tc2 \tc3 \tc4 \tHC \tRC \tIC \t E" << endl;
    cout << (isBusy ? "busy" : "free");
    cout << "\t" << this->lambda_;
    cout << "\t" << this->mu_;
    cout << "\t" << this->c1_;
    cout << "\t" << this->c2_;
    cout << "\t" << this->c3_;
    cout << "\t" << this->c4_;
    cout << "\t" << this->handlingCounter_;
    cout << "\t" << this->rejectingCounter_;
    cout << "\t" << this->idleCounter_;

    if (ticks == this->simulatedTicks_) {
        double efficienty = (this->c1_ - this->c2_) * 
                            (double)this->handlingCounter_ - 
                            this->c3_ * (double)this->rejectingCounter_ -
                            this->c4_ * (double)this->idleCounter_;

        cout << "\t" << efficienty;
    }
    cout << endl;
}

void QueuingSystem::runSimulate()
{
    int elapsedTicksFromLastRequest  = 0;
    int elapsedTicksFromLastHandling = 0;
    bool isSystemFree = true;

    for (auto i = 0; i < this->simulatedTicks_; ++i) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        elapsedTicksFromLastRequest++;

        // Generate incoming requests
        double p_lambda = this->calcIncomingProbability(elapsedTicksFromLastRequest);
        double rand = dis(gen);
        if (rand < p_lambda) {
            if (isSystemFree) {
                isSystemFree = false;
            } else {
                this->rejectingCounter_++;
            }
            elapsedTicksFromLastRequest = 0;
        }

        if (isSystemFree) {
            this->idleCounter_++;
        }

        // Try to process requests
        if (!isSystemFree) {
            double p_mu = this->calcHandlingProbability(elapsedTicksFromLastHandling);
            rand = dis(gen);
            if (rand < p_mu) {
                isSystemFree = true;
                elapsedTicksFromLastHandling = 0;

                this->handlingCounter_++;
            }
            elapsedTicksFromLastHandling++;
        }

    }

    this->printStat(this->simulatedTicks_, isSystemFree);
    //cout << "*********************************************";
    //cout << "*********************************************" << endl;
}

int main(int argc, char const *argv[])
{
    QueuingSystem qs01(0.5, 0.001, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs02(0.5, 0.01, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs03(0.5, 0.02, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs04(0.5, 0.04, 1000, 1.0, 0.5, 0.15, 0.2);

    QueuingSystem qs0(0.5, 0.05, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs1(0.5, 0.10, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs2(0.5, 0.15, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs3(0.5, 0.20, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs4(0.5, 0.25, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs5(0.5, 0.30, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs6(0.5, 0.35, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs7(0.5, 0.40, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs8(0.5, 0.45, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs9(0.5, 0.50, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs10(0.5, 0.60, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs11(0.5, 0.70, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs12(0.5, 0.80, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs13(0.5, 0.90, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs14(0.5, 1.0, 1000, 1.0, 0.5, 0.15, 0.2);
    QueuingSystem qs15(0.5, 2.0, 1000, 1.0, 0.5, 0.15, 0.2);

    cout << "\nSS - System state"<< endl;
    cout << "λ  - incoming"<< endl;
    cout << "μ  - handling"<< endl;
    cout << "c1 - profit"<< endl;
    cout << "c2 - expenses"<< endl;
    cout << "c3 - loss"<< endl;
    cout << "c4 - idle"<< endl;
    cout << "HC - Handling Counter"<< endl;
    cout << "RC - Rejecting Counter"<< endl;
    cout << "IC - Idle Counter"<< endl;
    cout << "E  - Efficienty"<< endl << endl;
    cout << "\nSS \tλ \tμ \tc1 \tc2 \tc3 \tc4 \tHC \tRC \tIC \t E" << endl;

    qs01.runSimulate();
    qs02.runSimulate();
    qs03.runSimulate();
    qs04.runSimulate();

    qs0.runSimulate();
    qs1.runSimulate();
    qs2.runSimulate();
    qs3.runSimulate();
    qs4.runSimulate();
    qs5.runSimulate();
    qs6.runSimulate();
    qs7.runSimulate();
    qs8.runSimulate();
    qs9.runSimulate();
    qs10.runSimulate();
    qs11.runSimulate();
    qs12.runSimulate();
    qs13.runSimulate();
    qs14.runSimulate();
    qs15.runSimulate();

    return 0;
}
