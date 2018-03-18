#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>



#ifdef TEST_SIZE_1
enum { NELEMS = 1<<10 };
#endif /* SIZE_TEST_1 */

#ifdef TEST_SIZE_2
enum { NELEMS = 1<<16 };
#endif /* SIZE_TEST_2 */

#ifdef TEST_SIZE_3
enum { NELEMS = 1<<22 };
#endif /* SIZE_TEST_3 */

int main(void)
{
#ifdef TEST_CONF_1
    int threadsPerBlock = 512;
#endif /* TEST_CONF_1 */

#ifdef TEST_CONF_2
    int threadsPerBlock = 1024;
#endif /* TEST_CONF_2 */

    int blocksPerGrid = (NELEMS + threadsPerBlock - 1) / threadsPerBlock;

    printf("Vector size       is %d elements\n", NELEMS);
    printf("Threads per block is %d elements\n", threadsPerBlock);
    printf("Blocks per grid   is %d elements\n", blocksPerGrid);

    return 0;
}
