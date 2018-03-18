#include "io_file.h"

using namespace std;

int main(int argc, char const *argv[])
{
	set_file_name(argv[1]);  
    read_from_file(argv[1]);
    return 0;
}