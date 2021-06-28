#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAXLINE 128

void pgmread(char *filename, void *vx, int nx, int ny)
{ 
  FILE *fp;

  int nxt, nyt, i, j, t;
  char dummy[MAXLINE];
  int n = MAXLINE;

  char *cret;
  int iret;

  float *x = (float *) vx;

  fopen_s(&fp, filename, "r");

  if (fp == NULL)
  {
    fprintf(stderr, "pgmread: cannot open <%s>\n", filename);
    exit(-1);
  }
  
  cret = fgets(dummy, n, fp);
  cret = fgets(dummy, n, fp);

  iret = fscanf_s(fp,"%d %d",&nxt,&nyt);

  if (nx != nxt || ny != nyt)
  {
    fprintf(stderr,
            "pgmread: size mismatch, (nx,ny) = (%d,%d) expected (%d,%d)\n",
            nxt, nyt, nx, ny);
    exit(-1);
  }

  iret = fscanf_s(fp,"%d",&i);

  for (j=0; j<ny; j++)
  {
    for (i=0; i<nx; i++)
    {
      iret = fscanf_s(fp,"%d", &t);
      x[(ny-j-1)+ny*i] = t;
    }
  }

  fclose(fp);
}

void datread(char *filename, void *vx, int nx, int ny)
{
	FILE *fp;

	int nxt, nyt, i, j, t;

	float *x = (float *)vx;

	if (NULL == (fp = fopen(filename, "r")))
	{
		fprintf(stderr, "datread: cannot open <%s>\n", filename);
		exit(-1);
	}

	fscanf(fp, "%d %d", &nxt, &nyt);

	if (nx != nxt || ny != nyt)
	{
		fprintf(stderr,
			"datread: size mismatch, (nx,ny) = (%d,%d) expected (%d,%d)\n",
			nxt, nyt, nx, ny);
		exit(-1);
	}

	for (j = 0; j<ny; j++)
	{
		for (i = 0; i<nx; i++)
		{
			fscanf(fp, "%d", &t);
			x[(ny - j - 1)*nx + i] = t;
		}
	}

	fclose(fp);
}

void pgmwrite(char *filename, void *vx, int nx, int ny)
{
	FILE *fp;

	int i, j, k, grey;

	float xmin, xmax, tmp;
	float thresh = 255.0;

	float *x = (float *)vx;

	if (NULL == (fp = fopen(filename, "w")))
	{
		fprintf(stderr, "pgmwrite: cannot create <%s>\n", filename);
		exit(-1);
	}

	xmin = fabs(x[0]);
	xmax = fabs(x[0]);

	for (i = 0; i < nx*ny; i++)
	{
		if (fabs(x[i]) < xmin) xmin = fabs(x[i]);
		if (fabs(x[i]) > xmax) xmax = fabs(x[i]);
	}

	fprintf(fp, "P2\n");
	fprintf(fp, "# Written by pgmwrite\n");
	fprintf(fp, "%d %d\n", nx, ny);
	fprintf(fp, "%d\n", (int)thresh);

	k = 0;

	for (j = ny - 1; j >= 0; j--)
	{
		for (i = 0; i < nx; i++)
		{
			tmp = x[j*nx + i];

			if (xmin < 0 || xmax > thresh)
			{
				tmp = (int)((thresh*((fabs(tmp - xmin)) / (xmax - xmin))) + 0.5);
			}
			else
			{
				tmp = (int)(fabs(tmp) + 0.5);
			}

			grey = (int)(thresh * sqrt(tmp / thresh));

			fprintf(fp, "%3d ", grey);

			if (0 == (k + 1) % 16) fprintf(fp, "\n");

			k++;
		}
	}

	if (0 != k % 16) fprintf(fp, "\n");
	fclose(fp);
}
