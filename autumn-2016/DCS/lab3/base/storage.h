#ifndef _STORAGE_H
#define _STORAGE_H

#include <mutex>
#include <vector>

using namespace std;

class Room
{
    private:
        int idRoom_;
        int roomCapacity_;

    public:
        Room(int id);
        ~Room();

        void showInfo();

        int checkProduction(int needProduction);
        int getProduction(int needProduction);
        void fillRoom(int production);
};

class Storage
{
    private:
        vector<Room*> rooms_;
        mutex loadingAreaMutex_;

    public:
        Storage(int rooms);
        ~Storage();

        void lockLoadingArea();
        void unlockLoadingArea();

        int getProduction(int needProduction);
        void fillRooms(int production);

        void showStorage();
    
};

#endif /* _STORAGE_H */