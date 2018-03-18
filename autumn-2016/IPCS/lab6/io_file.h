#ifndef _OI_FILE_H_
#define _OI_FILE_H_

#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

void set_file_name(const char *file);
void write_to_file(std::string buffer);
void read_from_file(const char *file);

#endif /* _OI_FILE_H_ */