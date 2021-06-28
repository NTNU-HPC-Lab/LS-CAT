#pragma once
//#include "kernel.cu"
typedef struct Point
{
	float* coordinates;

}Point;

typedef struct Cluster
{
	int num_points_in_cluster; // optional
	float diameter;
	Point center;
	Point* points_in_cluster;

}Cluster;

const int parallel = 0;

void print_number_of_points_cluster(Cluster clusters);
void print_diamter(Cluster cluster);
void print_cluster_ditails(Cluster* all_clusters, int num_clusters);
void print_point_of_cluster(Cluster cluster, int num_coordinates);
Cluster* init_clusters(Cluster* all_clusters, Point* all_points, int num_clusters , int num_points);
void reset_points_and_diameter_in_clusters(Cluster* all_clusters, int num_clusters);
float* getVarsFromTXT();
Point* getPointsFromTXT(Point* all_points,int N, int n);
double calc_distance_between_2_coordinate(double coor_1, double coor_2, int square);
double calc_distance_between_2_clusters(Cluster cluster_1, Cluster cluster_2, int num_coordinates);
double get_distance_between_2_points(Point point_1,Point point_2, int num_coordinates);
void associate_points_to_clusters(Point* all_points, Cluster* all_clusters, int num_points, int num_clusters, int num_coordinates);
void recenter_single_cluster(Cluster* cluster, int num_coordinates);
void recenter_all_clusters(Cluster* all_clusters,int num_clusters, int num_coordinates);
double calc_diameter_of_single_cluster(Cluster* cluster, int num_coordinates);
void calc_diameter_of_all_clusters(Cluster* all_clusters,int num_clusters, int num_coordinates);
double calc_qm (Cluster* all_clusters,int num_clusters, int num_coordinates);
int calc_min(float* dist_arr, int num_clusters);
double sum_coordinates_arr(float* corrdinates, int num_coordinates);
float* get_diameter_of_all_clusters (Cluster* all_clusters, int num_clusters);
int* get_num_of_points_of_all_clusters (Cluster* all_clusters, int num_clusters);
bool check_termination_condition(Cluster* all_clusters, int num_clusters, int num_coordinates);
void print_logs(int var);
float* reset_arr(float* arr, int size);
void write_results_to_file(Cluster* all_clusters, int num_clusters, int num_coordinates, double qm);
void print_cluster_center(Cluster cluster, int num_coordinates);
void print_clusters_centers(Cluster* all_clusters, int num_clusters, int num_coordinates);
