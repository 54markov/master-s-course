#include <iostream>
#include <thread>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#include "customer.h"
#include "storage.h"
#include "loader.h"
#include "timer.h"

using namespace std;

int main(int argc, char const *argv[])
{
    signal(SIGALRM, signalhandler);

    try {
        if (argc != 4) {
            throw string("usage: " + string(argv[0]) + " [storage N] [custome K] [time to live T]");
        }
    } catch (string &err) {
        cerr << err << endl;
        return -1;
    }

    int N = atoi(argv[1]);
    int K = atoi(argv[2]);
    int T = atoi(argv[3]);

    cout << "storage     : " << N << endl;
    cout << "customer    : " << K << endl;
    cout << "time to live: " << T << endl;

    Timer timer(T);

    Storage mainStorage(N);
    
    Loader loader(&mainStorage);

    mainStorage.showStorage();

    vector<thread> customerThreads;

    createAndRunCustomers(&customerThreads, K, &mainStorage);

    loader.createAndRunLoader();

    while(1) {

    }

    return 0;
}