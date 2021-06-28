/* stdio.h: The ABC representation of standard C library stdio.
 * Based on C11 Standard.
 */
#ifndef __STDIO__
#define __STDIO__

/* Needed from stdarg.h: */

#include <stdarg.h>

/* Types */

typedef unsigned long int size_t;

typedef struct FILE FILE;

typedef int fpos_t;

/* Macros */

#ifndef NULL
#define NULL ((void*)0)
#endif
#define _IOFBF 1
#define _IOLBF 2
#define _IONBF 3
#define BUFSIZ 100
#define EOF (-100)
#define FOPEN_MAX 100
#define FILENAME_MAX 500
#define L_tmpnam 500
#define SEEK_CUR 1
#define SEEK_END 2
#define SEEK_SET 3
#define TMP_MAX 100
//#define stdin (FILE*)0
//#define stdout (FILE*)1
//#define stderr (FILE*)2

/* external variables */

extern FILE * stdout;
extern FILE * stdin;
extern FILE * stderr;

/* Functions */

int remove(const char *filename);
int rename(const char *old, const char *new);
FILE *tmpfile(void);
char *tmpnam(char *s);
int fclose(FILE *stream);
int fflush(FILE *stream);
FILE *fopen(const char * restrict filename,
     const char * restrict mode);
FILE *freopen(const char * restrict filename,
     const char * restrict mode,
     FILE * restrict stream);
void setbuf(FILE * restrict stream,
     char * restrict buf);
int setvbuf(FILE * restrict stream,
     char * restrict buf,
     int mode, size_t size);
int fprintf(FILE * restrict stream,
     const char * restrict format, ...);
int fscanf(FILE * restrict stream,
     const char * restrict format, ...);
int printf(const char * restrict format, ...);
int scanf(const char * restrict format, ...);
int snprintf(char * restrict s, size_t n,
     const char * restrict format, ...);
int sprintf(char * restrict s,
     const char * restrict format, ...);
int sscanf(const char * restrict s,
     const char * restrict format, ...);
int vfprintf(FILE * restrict stream,
     const char * restrict format, va_list arg);
int vfscanf(FILE * restrict stream,
     const char * restrict format, va_list arg);
int vprintf(const char * restrict format, va_list arg);
int vscanf(const char * restrict format, va_list arg);
int vsnprintf(char * restrict s, size_t n,
     const char * restrict format, va_list arg);
int vsprintf(char * restrict s,
     const char * restrict format, va_list arg);
int vsscanf(const char * restrict s,
     const char * restrict format, va_list arg);
int fgetc(FILE *stream);
char *fgets(char * restrict s, int n,
     FILE * restrict stream);
int fputc(int c, FILE *stream);
int fputs(const char * restrict s,
     FILE * restrict stream);
int getc(FILE *stream);
int getchar(void);
int putc(int c, FILE *stream);
int putchar(int c);
int puts(const char *s);
int ungetc(int c, FILE *stream);
size_t fread(void * restrict ptr,
     size_t size, size_t nmemb,
     FILE * restrict stream);
size_t fwrite(const void * restrict ptr,
     size_t size, size_t nmemb,
     FILE * restrict stream);
int fgetpos(FILE * restrict stream,
     fpos_t * restrict pos);
int fseek(FILE *stream, long int offset, int whence);
int fsetpos(FILE *stream, const fpos_t *pos);
long int ftell(FILE *stream);
void rewind(FILE *stream);
void clearerr(FILE *stream);
int feof(FILE *stream);
int ferror(FILE *stream);
void perror(const char *s);

/* Optional Extensions */

#define __STDC_WANT_LIB_EXT1__ 1
#define L_tmpnam_s 100
#define TMP_MAX_S 100
typedef int errno_t;
typedef size_t rsize_t;
errno_t tmpfile_s(FILE * restrict * restrict streamptr);
errno_t tmpnam_s(char *s, rsize_t maxsize);
errno_t fopen_s(FILE * restrict * restrict streamptr,
     const char * restrict filename,
     const char * restrict mode);
errno_t freopen_s(FILE * restrict * restrict newstreamptr,
     const char * restrict filename,
     const char * restrict mode,
     FILE * restrict stream);
int fprintf_s(FILE * restrict stream,
     const char * restrict format, ...);
int fscanf_s(FILE * restrict stream,
     const char * restrict format, ...);
int printf_s(const char * restrict format, ...);
int scanf_s(const char * restrict format, ...);
int snprintf_s(char * restrict s, rsize_t n,
     const char * restrict format, ...);
int sprintf_s(char * restrict s, rsize_t n,
     const char * restrict format, ...);
int sscanf_s(const char * restrict s,
     const char * restrict format, ...);
int vfprintf_s(FILE * restrict stream,
     const char * restrict format,
     va_list arg);
int vfscanf_s(FILE * restrict stream,
     const char * restrict format,
     va_list arg);
int vprintf_s(const char * restrict format,
     va_list arg);
int vscanf_s(const char * restrict format,
     va_list arg);
int vsnprintf_s(char * restrict s, rsize_t n,
     const char * restrict format,
     va_list arg);
int vsprintf_s(char * restrict s, rsize_t n,
     const char * restrict format,
     va_list arg);
int vsscanf_s(const char * restrict s,
     const char * restrict format,
     va_list arg);
char *gets_s(char *s, rsize_t n);

#endif
