#include "storage.h"

#include <iostream>
#include <vector>
#include <random>

#include <ctime>
#include <ratio>
#include <chrono>

using namespace std;

Room::Room(int id)
{
    auto seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 40);

    this->idRoom_       = id;
    this->roomCapacity_ = distribution(generator);
}

Room::~Room() {}

void Room::showInfo()
{
    cout << "| +----------------" << endl;
    cout << "| |Room     : " << this->idRoom_ << endl;
    cout << "| |Capacity : " << this->roomCapacity_ << endl;
    cout << "| +----------------" << endl;
}

int Room::getProduction(int needProduction)
{
    if (this->roomCapacity_ == 0) {
        return 0;
    }

    if ((needProduction - this->roomCapacity_) > 0) {
        auto tmp = this->roomCapacity_;
        this->roomCapacity_ = 0;
        return tmp;
    } else {
        this->roomCapacity_ = this->roomCapacity_ - needProduction;
        return needProduction;
    }
}

int Room::checkProduction(int needProduction)
{
    return this->roomCapacity_;
}

void Room::fillRoom(int amount)
{
    if (this->roomCapacity_ == 0) {
        this->roomCapacity_ = amount;
    }
}



Storage::Storage(int rooms)
{
    for (auto i = 0; i < rooms; ++i) {
        Room *newRoom = new Room(i);
        this->rooms_.push_back(newRoom);
    }
}

Storage::~Storage() {}

void Storage::lockLoadingArea() { this->loadingAreaMutex_.lock(); }
void Storage::unlockLoadingArea() { this->loadingAreaMutex_.unlock(); }

void Storage::showStorage()
{
    cout << "+---------------------+" << endl;
    cout << "| Storage:            |" << endl;
    cout << "+---------------------+" << endl;
    
    for (auto i = 0; i < rooms_.size(); ++i) {
        rooms_[i]->showInfo();
    }
    
    cout << "+---------------------+" << endl;
}

/*
int Storage::getProduction(int needProduction)
{
    int takenProduction = 0;
    int productionFlag = 0;
    
    for (auto i = 0; i < rooms_.size(); ++i) {
        takenProduction += rooms_[i]->checkProduction(needProduction);

        if (needProduction == takenProduction) {
            takenProduction = 0;
            productionFlag = 1;
            cout << __func__ << endl;
            break;
        }
    }

    if (productionFlag) {
        for (auto i = 0; i < rooms_.size(); ++i) {
            auto tmp = rooms_[i]->getProduction(needProduction);

            takenProduction += tmp;
            needProduction -= tmp;

            if (needProduction <= 0) {
                break;
            }
        }
        return needProduction;
    }

    return 0;
}
*/
int Storage::getProduction(int needProduction)
{
    int takenProduction = 0;
    
        for (auto i = 0; i < rooms_.size(); ++i) {
        auto tmp = rooms_[i]->getProduction(needProduction);

        takenProduction += tmp;
        needProduction -= tmp;
        

        //cout << "tmp            " << tmp << endl;
        //cout << "needProduction " << needProduction << endl << endl;

        if (needProduction <= 0) {
            break;
        }
    }

    for (auto i = 0; i < rooms_.size(); ++i) {
        auto tmp = rooms_[i]->getProduction(needProduction);

        takenProduction += tmp;
        needProduction -= tmp;
        

        //cout << "tmp            " << tmp << endl;
        //cout << "needProduction " << needProduction << endl << endl;

        if (needProduction <= 0) {
            break;
        }
    }

    //this->showStorage();

    //cout << "takenProduction " << takenProduction << endl;

    return takenProduction;
}

void Storage::fillRooms(int production)
{
    for (auto i = 0; i < rooms_.size(); ++i) {
        rooms_[i]->fillRoom(production);
    }

    cout << "Loader" << endl << endl;
    this->showStorage();
    cout << endl << endl;
}