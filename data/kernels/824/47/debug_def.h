#pragma once
#include <assert.h>

#define SYM_TO_STR_(x) #x
#define SYM_TO_STR(x)  SYM_TO_STR_(x)
#define __SRC_LOC__    "\n\n(" SYM_TO_STR(__FILE__) ", #" SYM_TO_STR(__LINE__) ")\n"

#ifdef NDEBUG
	#define INSIST(exp) { bool expEval = (exp); if(expEval == false) fprintf(stderr, "%sASSERTION failed: (%s).\n", __SRC_LOC__, #exp); }
	#undef  assert
	#define assert INSIST
#else
	#define INSIST assert
#endif

#define ALERT(...) fprintf(stderr, "%s", __SRC_LOC__); fprintf (stderr, __VA_ARGS__)

#define TOUCH(x)        ;{(x);};
#define PRINT_STR(x)    printf(#x ": %s\n",          (const char *)           (x));
#define PRINT_STRING(x) printf(#x ": %s\n",          (const char *)           (x).c_str());
#define PRINT_CHAR(x)   printf(#x ": %c\n",          (char)                   (x));
#define PRINT_INT(x)    printf(#x ": %i\n",          (int)                    (x));
#define PRINT_UINT(x)   printf(#x ": %u\n",          (unsigned int)           (x));
#define PRINT_SIZE(x)   printf(#x ": %zu\n",         (size_t)                 (x));
#define PRINT_LLINT(x)  printf(#x ": %lli\n",        (long long int)          (x));
#define PRINT_ADDR(x)   printf(#x ": %016p\n",       (void*)                  (x));
#define PRINT_FLOAT(x)  printf(#x ": %f\n",          (float)                  (x));
#define PRINT_FRAC(x)   printf(#x ": %.2f\n",        (float)                  (x));
#define PRINT_DOUBLE(x) printf(#x ": %lf\n",         (double)                 (x));
#define PRINT_SCI(x)    printf(#x ": %e\n",          (double)                 (x));
#define PRINT_BOOL(x)   printf(#x ": %s\n",          (bool(x) == false) ? "false" : "true");
#define PRINT_PT(pt)    printf(#pt": %.2f,%.2f\n",   (float) (pt).x(), (float) (pt).y());
#define PRINT_LOC()     PRINT_STR(__SRC_LOC__)
#define PRINT_OBJ(x)    {                        \
                            printf(#x ":\n");    \
                            (x).Dump();          \
						}
#define PRINT_M_OBJ(x)  {                        \
                            printf(#x ":\n");    \
                            MathHelper::Dump(x); \
						}
#define PRINT_INTS(x)   {                                                                      \
                            printf(#x ": [ ");                                                 \
                            for(size_t i = 0; i < (x).size(); i++) printf("%i ", int((x)[i])); \
                            printf("]\n");                                                     \
						}
#define PRINT_PTS(pts)  {                                                                                                                   \
                            printf(#pts ": [ ");                                                                                            \
                            for (size_t i = 0; i < (pts).size(); i++) printf("%.2f,%.2f ", (float) ((pts)[i]).x(), (float) ((pts)[i]).y()); \
                            printf("]\n");                                                                                                  \
						}
#define PRINT_STRINGS(x){                                                                              \
                            printf(#x ":\n");                                                          \
                            for(size_t i = 0; i < (x).size(); i++) printf("\t%s\n", ((x)[i]).c_str()); \
						}

#define INSIST_RANGE(x, xMin, xMax) {                          \
								 	    INSIST((x) >= (xMin)); \
									    INSIST((x) <= (xMax)); \
                                    }

#define IN_RANGE(x, xMin, xMax) (((x) >= (xMin)) && ((x) <= (xMax)))
