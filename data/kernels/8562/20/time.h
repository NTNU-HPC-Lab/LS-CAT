/*
 *  timing.h
 *  
 *
 *
 */

#ifndef TIMING_H
#define TIMING_H

#include <sys/time.h>
#include <time.h>

/* Subtract the `struct timeval' value 'then' from 'now',
   returning the difference as a float representing seconds
   elapsed.
*/
float elapsedTime(struct timeval now, struct timeval then);
float timeCost(clock_t start, clock_t end);
double currentTime();

#endif

