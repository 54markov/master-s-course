#ifndef _CUSTOMER_H
#define _CUSTOMER_H

#include <thread>

#include "storage.h"

class Customer
{
    private:
        int idCustomer;
        int needProduction;  // random generating numbers from 1 to 1000
        int takenProduction; // incrementing value

        Storage *mainStorage;
    public:
        Customer(int id, int production, Storage *storage);
        ~Customer();

        int getProductionFromStorage();
        void runCustomer();

        void showInfo();
};

void createAndRunCustomers(vector<thread> *customerThreads, int amount, Storage* mainStorage);
void createAndRunCustomer(int amount, Storage* mainStorage);

#endif /* _CUSTOMER_H */