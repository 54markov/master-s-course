/*
 * Разработайте программу, реализующее псевдопараллельное выполнение двух функций: 
 * одна из которых непрерывно выводит на экран символ A, 
 * а другая непрерывно на экран выводит символ B. 
 * Переключение между выполнением функций должно осуществляться 
 * раз в три секунды по сигналу от таймера.
 */

#include <stdio.h>
#include <signal.h>
#include <sys/time.h>

#include <unistd.h>

void print_a() { printf ("a"); fflush(stdout); }
void print_b() { printf ("b"); fflush(stdout); }

void (*call_back)();

void signalhandler(int signo)
{ 
    if (call_back == &print_a) {
        call_back = &print_b;
    } else {
        call_back = &print_a;
    }
}

int main(int argc, char const *argv[])
{
    struct itimerval nval, oval;

    signal(SIGALRM, signalhandler);

    nval.it_interval.tv_sec  = 3; // interval 
    nval.it_interval.tv_usec = 0;

    nval.it_value.tv_sec     = 3; // time until next expiration
    nval.it_value.tv_usec    = 0;

    setitimer(ITIMER_REAL, &nval, &oval);

    call_back = &print_a;

    while(1) { 
        (*call_back)();
        sleep(1);
    }

    return (0);
}
