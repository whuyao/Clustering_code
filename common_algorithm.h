#ifndef COMMON_ALGORITHM_H_FILE_20171107
#define COMMON_ALGORITHM_H_FILE_20171107

#include <vector> 
#include "alglib/statistics.h"
#include "alglib/dataanalysis.h"
#include "alglib/alglibmisc.h"

using namespace alglib;
using namespace std;
//存放元组的属性信息  
//存储每条数据记录，第一个位置存放记录编号，第2到dimNum+1个位置存放实际元素 
typedef vector<double> Tuple;

//dynamic time warpping distance
double DTW_distance(double* pArr1, double* pArr2, int nCount);
double DTW_distance(real_1d_array pArr1, real_1d_array pArr2);

//kmeans初始化，寻找最佳点位
bool DTW_findInitialCenters(vector<Tuple> origin_tuples, Tuple*& means, float*& pContribution, int nRealClusterNum, float max_gap_dist = 0.1f, int nClusterNumTimes = 2, int nRandParts = 20);

//DTW kmeans
/************************************************************************
tuples: input pre-clustering dataset
nClusterNum: num of dest-clusters
nMaxiterations: the maximum num of iterations
dMaxExpectations: the maximum expectations to stop iterations
distype: distance type -1/-2
clusters: output clusters result
isRndInit: initialization parameters, use random number to init clustering center
max_gap_dist: initialization parameters, the maximum distance between farest pt and center pt (for DTW)
nClusterNumTimes: max cluster
nRandParts: num of random split the dataset to several parts
minInitContribution: the minimum init contribution value
************************************************************************/
void DTW_KMeans(vector<Tuple>& tuples, int nClusterNum, int nMaxIteraions, double dMaxExpectation, int disttype, vector<Tuple>*& clusters, bool isRndInit, float max_gap_dist, int nClusterNumTimes, int nRandParts, const char* means_filename, float minInitContribution, const char *distanceFileName);
bool outputFile(vector<Tuple>* clusters, int nClusterNum, char* outputfilename);
bool silhouette(vector<Tuple>& tuples, vector<Tuple>* clusters, int nClusterNum, int ndisttype, float*& pSilValues, float& aveSilValue, const char* distanceFileName);
bool silhouette_output(vector<Tuple>& tuples, vector<Tuple>* clusters, int nClusterNum, int ndisttype, float*& pSilValues, float& aveSilValue, const char* outputfilename, const char* distanceFileName);

//tranditional kmeans
/*
DistType-   distance function:
*  0    Chebyshev distance  (L-inf norm)
*  1    city block distance (L1 norm)
*  2    Euclidean distance  (L2 norm)
* 10    Pearson correlation:
dist(a,b) = 1-corr(a,b)
* 11    Absolute Pearson correlation:
dist(a,b) = 1-|corr(a,b)|
* 12    Uncentered Pearson correlation (cosine of the angle):
dist(a,b) = a'*b/(|a|*|b|)
* 13    Absolute uncentered Pearson correlation
dist(a,b) = |a'*b|/(|a|*|b|)
* 20    Spearman rank correlation:
dist(a,b) = 1-rankcorr(a,b)
* 21    Absolute Spearman rank correlation
dist(a,b) = 1-|rankcorr(a,b)|
*/
bool TRAN_KMeans(real_2d_array trainingdata, real_1d_array dataids, const char* outputfilename, int nClassCount = 5, int ndistType = 2);

#endif
