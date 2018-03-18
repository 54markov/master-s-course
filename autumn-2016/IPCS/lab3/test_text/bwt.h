#ifndef _BWT_H_
#define _BWT_H_

#include <string>

using namespace std;

bool pairCompare(const std::pair<string, string>& firstElem, const std::pair<string, string>& secondElem);
string BWT(string str);
void reverseBWT(string key);

#endif /* _BWT_H_ */