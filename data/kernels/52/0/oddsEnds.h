#ifndef ODDSANDENDS_H
#define ODDSANDENDS_H

#define ASSERT(x, msg, retcode) \
    if (!(x)) \
    { \
        cout << msg << " " << __FILE__ << ":" << __LINE__ << endl; \
        return retcode; \
    }

// Forward declarations
int test_basic();
int test_pinned();
int test_zerocopy();
int default_test();
int test_UVA();
int test_uniformMem();


void sequence_cpu(int *h_ptr, int length);


#endif /* ODDSANDENDS_H */
