#ifndef __STATS_H__
#define __STATS_H__
#include <time.h>

#define P_COUNT 100
extern unsigned int frame_count;

static inline unsigned long getTimeInUs()
{
        struct timespec tm;
        clock_gettime(CLOCK_REALTIME, &tm);
        return (unsigned long)(tm.tv_sec*1000*1000*1000+tm.tv_nsec);
}
static inline unsigned long getTimeInMicroS()
{
        struct timespec tm;
        clock_gettime(CLOCK_REALTIME, &tm);
        return (unsigned long)(tm.tv_sec*1000*1000+tm.tv_nsec/1000);
}

static inline void sumFrameStats(unsigned long startTime, unsigned long *delta, unsigned int frameCount, const char * pName)
{
	unsigned long curTime=getTimeInMicroS();
	*delta+=curTime-startTime;
	if (frameCount%P_COUNT==0) {
	printf("%s %u frames takes %lu msec. start at %lu, print at %lu\n"
		, pName, frameCount, (*delta)/1000, startTime, curTime);
		*delta=0;
	}
}
#endif
