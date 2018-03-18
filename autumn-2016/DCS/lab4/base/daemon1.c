#include <unistd.h>

int main(void)
{
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    setsid();   // assing new sid

    for (int i = 0; i < 30; i++) {
        sleep(1);
    }

    return 0;
}
