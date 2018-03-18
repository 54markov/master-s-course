#include "customer.h"

#include <iostream>

#include <thread>
#include <random>

#include <ctime>
#include <ratio>
#include <chrono>

using namespace std;

void createAndRunCustomers(vector<thread> *customerThreads, int amount, Storage* mainStorage)
{
    for (auto i = 0; i < amount; ++i) {
        customerThreads->push_back(std::thread(createAndRunCustomer, i, mainStorage));
    }
}

void createAndRunCustomer(int id, Storage* mainStorage)
{
    auto seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 1000);

    Customer localCustomer(id, distribution(generator), mainStorage);

    localCustomer.runCustomer();
}

Customer::Customer(int id, int production, Storage *storage)
{
    this->idCustomer      = id;
    this->needProduction  = production;
    this->takenProduction = 0;
    this->mainStorage     = storage;
}

Customer::~Customer() {}

int Customer::getProductionFromStorage()
{
    this->mainStorage->lockLoadingArea();

    this->takenProduction += mainStorage->getProduction((this->needProduction - this->takenProduction));

    this->mainStorage->unlockLoadingArea();
}

void Customer::runCustomer()
{
    cout << "Customer " << this->idCustomer << " need " << this->needProduction << endl;
    
    while (this->needProduction != this->takenProduction) {
        this->getProductionFromStorage();
        this->showInfo();
        cout << "Customer " << this->idCustomer << " wait for production" << endl;
        this_thread::sleep_for(chrono::seconds(5));
    }

    cout << "Customer " << this->idCustomer << " got all production" << endl;
}

void Customer::showInfo()
{
    cout << "***************************************" << endl;
    cout << "*Customer " << this->idCustomer << endl;
    cout << "*Need     " << this->needProduction << endl;
    cout << "*Take     " << this->takenProduction << endl;
    cout << "***************************************" << endl;
}