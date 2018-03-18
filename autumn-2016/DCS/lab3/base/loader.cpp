#include "loader.h"

Loader::Loader(Storage* storage) 
{
    this->mainStorage_ = storage;
}

Loader::~Loader() {}

void Loader::createAndRunLoader()
{
    while (1) {
        this->mainStorage_->lockLoadingArea();

        this->mainStorage_->fillRooms(100);

        this->mainStorage_->unlockLoadingArea();

        this_thread::sleep_for(chrono::seconds(5));
    }
}
