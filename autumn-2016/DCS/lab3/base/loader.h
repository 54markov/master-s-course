#ifndef _LOADER_H
#define _LOADER_H

#include <thread>

#include "storage.h"

class Loader
{
    private:
        Storage* mainStorage_;

    public:
        Loader(Storage* storage);
        ~Loader();

        void createAndRunLoader();
};

#endif /* _LOADER_H */
