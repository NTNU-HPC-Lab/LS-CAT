#ifndef SETTINGS_H
#define SETTINGS_H

/*****************************************************************
// Camera Section
*****************************************************************/
// DVS camera resolution
#define DVS_RESOLUTION_WIDTH 128
#define DVS_RESOLUTION_HEIGHT 128

/*****************************************************************
// GUI
*****************************************************************/
// GUI refresh rate
#define GUI_RENDERING_FPS 20
// Refreshrate of statistics
#define GUI_STAT_UPDATE_FPS 5

/*****************************************************************
// CUDA
*****************************************************************/
// Threads in a single block
#define THREADS_PER_BLOCK 64
// Maximum number of events in shared memory
#define MAX_SHARED_GPU_EVENTS 128

/*****************************************************************
// OpticFlow estimator
// Additional filter settings located in filtersettings.cpp
*****************************************************************/
// Spatial resolution of garbor filter (odd value)
#define FILTER_SPATIAL_SIZE_PX 11
// Temporal resolution of time function between 0 and TEMPORAL_END
#define FILTER_TEMPORAL_RES 20
// End value x of temporal function t(x)
#define FILTER_TEMPORAL_END 0.7f
#define FLOW_DEFAULT_MIN_ENERGY_THRESHOLD 0.6f
// Interpolation modes
// 0: No interpolation
// 1: No Speed interpolation, only orientation interpolation
// 2: Simultanious speed and orientation interpolation with
//    exponential weighting of energies
#define FLOW_INTERPOLATION_MODE 0
/*
// The maximum number of events per second (per motion energy)
// Additional events are skipped
#define FLOW_MAX_EVENTS_PER_SEC 20000
// Low pass filter coefficient to compute averaged events per slot
// Values between 0 - 1, greater values produce faster adaption
#define FLOW_SKIPPING_LOW_PASS_FILTER_COEFF (0.3)
*/

/*****************************************************************
// Pushbot
*****************************************************************/
// Pushbot commands
#define CMD_SET_TIMESTAMP_MODE "!E4\n"      // Do not change streaming mode !
#define CMD_ENABLE_EVENT_STREAMING "E+\n"
#define CMD_DISABLE_EVENT_STREAMING "E-\n"
#define CMD_ENABLE_MOTORS "!M+\n"
#define CMD_DISABLE_MOTORS "!M-\n"
#define CMD_SET_VELOCITY "!MV%1=%2\n"
#define CMD_RESET_BOARD "R\n"
#define CMD_UART_ENABLE_ECHO_MODE "!U1\n"
//#define CMD_UART_DISABLE_ECHO_MODE "!U0\n"
// Refresh rate of optic flow processing
#define PUSH_BOT_PROCESS_FPS 20
// Motor Velocity minimum for pid control (0-100)
#define PUSHBOT_VELOCITY_MIN 1
// Motor Velocity maximum for pid control
#define PUSHBOT_VELOCITY_MAX 40
// Pushbot motor number left
#define PUSHBOT_MOTOR_LEFT 0
// Pushbot motor number right
#define PUSHBOT_MOTOR_RIGHT 1
// Default speed
#define PUSHBOT_VELOCITY_DEFAULT 20
// Default PID values for pushbot PID controller
#define PUSHBOT_PID_P_DEFAULT 0.2
#define PUSHBOT_PID_I_DEFAULT 0.4
#define PUSHBOT_PID_D_DEFAULT 0.005
// Maximum absolute integrated error
#define PUSHBOT_PID_MAX_ESUM 500.f

// Minimum detection energy
// Summed energy over an image half hast to be greater as the value below
// Otherwise not steering signal is generated
#define PUSHBOT_MIN_DETECTION_ENERGY 50

/*****************************************************************
// Debug Section
*****************************************************************/
// Insert marks for Nvidia's visual profiler
#define DEBUG_INSERT_PROFILER_MARKS

/*****************************************************************
// Other
*****************************************************************/
// Maximum wait time for stopping a thread
#define THREAD_WAIT_TIME_MS 300
// Convolution buffers per filter orientation
// DO NOT CHANGE
#define FILTERS_PER_ORIENTATION 2

/*****************************************************************
// Macros
*****************************************************************/
#define CLAMP(v,mn,mx) qMin(mx,qMax(mn,v))
#define DEG2RAD(d) ((d)*M_PI/180.0)
#define RAD2DEG(r) ((r)/M_PI*180.0)
#define SIGN(T) (((0) < (T)) - ((T) < (0)))

#ifdef QT_DEBUG
#define PRINT_DEBUG(msg) qDebug(msg)
#define PRINT_DEBUG_FMT(format, ...) qDebug(format,__VA_ARGS__)
#else
#define PRINT_DEBUG(msg) {}
#define PRINT_DEBUG_FMT(format, ...) {}
#endif
#endif // SETTINGS_H
