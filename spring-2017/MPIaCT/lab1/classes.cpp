#include "classes.h"

Process::Process(int _id, int _tick, vector<char> *_array)
{
    id_          = _id;
    tick_        = _tick;
    globalArray_ = _array;

    local_tick_  = 0;
};

Process::~Process() {};

int Process::runProcess()
{
    if (tick_ == 0) {
        tick_--;
        return TICK_COMPLETE;
    } else if (tick_ > 0) {
        switch (id_)
        {
            case READER_1:
                localArray_.push_back(globalArray_->at(local_tick_));
                break;

            case READER_2:
            {
                if (checkChar(globalArray_->at(local_tick_)) == VOWEL_CHAR) {
                    localArray_.push_back(globalArray_->at(local_tick_));
                }
                break;
            }

            case READER_3:
            {
                if (checkChar(globalArray_->at(local_tick_)) == CONSONANT_CHAR) {
                    localArray_.push_back(globalArray_->at(local_tick_));
                }
                break;
            }

            case WRITER_1:
                globalArray_->at(local_tick_) = '*';
                break;

            default:
                cerr << "...";
                break;
        }

        local_tick_++;
        tick_--;
        return TICK_UNCOMPLETE;
    } else {
        return TICK_END;
    }
    return TICK_UNCOMPLETE;
}

int Process::checkChar(char ch)
{
    std::vector<char> vowel     = { 'A', 'E', 'I' };
    std::vector<char> consonant = { 'B', 'C', 'D', 'F', 'G', 'H', 'J' };

    switch (id_)
    {
        case READER_2:
        {
            for (auto i = 0; i < (int)vowel.size(); i++) {
                if (vowel[i] == ch)
                    return VOWEL_CHAR;
            }
            break;
        }

        case READER_3:
        {
            for (auto i = 0; i < (int)consonant.size(); i++) {
                if (consonant[i] == ch)
                    return CONSONANT_CHAR;
            }
            break;
        }

        default:
            cerr << "...";
            break;
    }

    return -1;
}

void Process::printStat()
{
    for (auto i: localArray_) {
        cout << i << " ";
    }
}

void Process::flushOut()
{
    tick_       = TICKS;
    local_tick_ = 0;
    localArray_.clear();
}