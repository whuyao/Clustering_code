#include <QtCore>
#include <iostream>
#include "common_algorithm.h"
using namespace std;

int main(int argc, char *argv[])
{
	cout << "programmed by Yaoyao & Maike & Kampau. Copyright 2016." << endl;
	cout << "any problem please contact: whuyao@foxmail.com" << endl;

	if (argc < 13)
	{
		cout << "please input data like this: " << endl;
		//TecentPopClustering.exe ./data/4401_timeline_pop.csv ./data/4401_timeline_pop_result_class_2.csv 2 8 1.0 -2 1 0 0.1 5 20 0.4
		cout << ">> TecentPopClustering.exe input_filename output_filename cluster_num max_iterations expectation_silhouette distance_type is_normalization(0=No/1=Yes) use_rand_init(0=No/1=Yes) max_DT_FT_gap_dist cluster_num_times AP_init_parts_num min_init_contribution" << endl;
		cout << "distance type = " << endl;
		cout << "\t-3 COS Distance (KMEDOIDS)" << endl;
		cout << "\t-2 Dynamic Time Wrapping Distance (KMEDOIDS)" << endl;
		cout << "\t-1 Normal Euclidean Distance" << endl;
		cout << "\t0 Chebyshev distance (L-inf norm)" << endl;
		cout << "\t1 city block distance (L1 norm)" << endl;
		cout << "\t2 Euclidean distance (L2 norm)" << endl;
		cout << "\t10 Pearson correlation: dist(a,b) = 1-corr(a,b)" << endl;
		cout << "\t11 Absolute Pearson correlation: dist(a,b) = 1-|corr(a,b)|" << endl;
		cout << "\t12 Uncentered Pearson correlation (cosine of the angle): dist(a,b) = a'*b/(|a|*|b|)" << endl;
		cout << "\t13 Absolute uncentered Pearson correlation: dist(a,b) = |a'*b|/(|a|*|b|)" << endl;
		cout << "\t20 Spearman rank correlation: dist(a,b) = 1-rankcorr(a,b)" << endl;
		cout << "\t21 Absolute Spearman rank correlation: dist(a,b) = 1-|rankcorr(a,b)|" << endl;
		cout << "exit." << endl;
		return -1;
	}

	cout << "programmed by YAOYAO & Maike. Copyright 2016." << endl;
	cout << "any problem please contact: whuyao@foxmail.com" << endl;

	const char* inputfilename = argv[1];//"./data/Cluster.csv";
	char* outputfilename = argv[2];//"./data/cluster_result.csv";
	int cluster_num = atoi(argv[3]);
	int nMaxIterations = atoi(argv[4]);
	double dMinError = atof(argv[5]);
	int ndisttype = atoi(argv[6]);
	int is_normalization = atoi(argv[7]);

	bool isUseRand = false;
	if (atoi(argv[8]) <= 0) isUseRand = false;
	else isUseRand = true;

	float max_gap_dist = atof(argv[9]);
	int nClusterNumTimes = atoi(argv[10]);
	int nRandParts = atoi(argv[11]);
	float dMininInitContribution = atof(argv[12]);


// 	const char* inputfilename = "./data/guangzhou_workday_24hours.csv";
// 	char* outputfilename = "./data/guangzhou_workday_24hours_cluster_result.csv";
// 	int cluster_num = 6;
// 	int nMaxIterations = 8;
// 	double dMinError = 1.0;
// 	int ndisttype = -3;
// 	int is_normalization = 2;
// 
// 	bool isUseRand = false;
// 
// 	float max_gap_dist = 0.1;
// 	int nClusterNumTimes = 5;
// 	int nRandParts = 20;
// 	float dMininInitContribution = 0.05;

	QFileInfo _info(outputfilename);
	QString eva_filename = _info.absolutePath() + "/" + _info.baseName() + "_silhouette.txt";
	QString means_filename = _info.absolutePath() + "/" + _info.baseName() + "_means.txt";

	cout << "******************** KMEANS ********************" << endl;
	cout << "input file name = " << inputfilename << endl;
	cout << "output file name = " << outputfilename << endl;
	cout << "silhouette file name = " << eva_filename.toStdString().data() << endl;
	cout << "means file name = " << means_filename.toStdString().data() << endl;
	cout << "clusters num = " << cluster_num << endl;
	cout << "max iterations num = " << nMaxIterations << endl;
	cout << "min gap error = " << dMinError << endl;
	cout << "distance type = " << ndisttype << endl;
	cout << "normalization = " << is_normalization << endl;
	cout << "rand init method = " << isUseRand << endl;
	cout << "max gap between DTW-center & farest Pt = " << max_gap_dist << endl;
	cout << "cluster init times = " << nClusterNumTimes << endl;
	cout << "AP init parts num = " << nRandParts << endl;
	cout << "min init contribution = " << dMininInitContribution << endl;

	//读入数据
	vector<Tuple> tuples;
	tuples.clear();

	QFile _file(inputfilename);
	if (!_file.open(QIODevice::ReadOnly))
	{
		cout << "open file failed." << endl;
		return -1;
	}

	QFileInfo _dis_info(inputfilename);
	QString distance_filename = _dis_info.absolutePath() + "/" + _dis_info.completeBaseName() + ".dat";
	cout << distance_filename.toStdString() << endl;

	QTextStream _in(&_file);
	QString smsg = _in.readLine();
	QStringList slist = smsg.split(",");
	int nFeaturesCount = slist.size() - 1;

	cout << "Features Count = " << nFeaturesCount << endl;

	while (!_in.atEnd())
	{
		smsg = _in.readLine();
		slist = smsg.trimmed().split(",");

		if (slist.size() < nFeaturesCount + 1) continue;

		Tuple tuple;
		tuple.push_back(slist[0].trimmed().toInt());

		double dSum = 0;
		for (int i = 1; i < slist.size(); i++)
		{
			tuple.push_back(slist[i].toDouble());
			dSum += slist[i].toDouble();
		}
		//如果采用余弦距离，出现0向量时直接退出程序
		if (ndisttype == -3 && dSum == 0)
		{
			cout << "There are some zero vector, please remove zero vector and try again." << endl;
			exit(0);
		}

		tuples.push_back(tuple);
	}

	cout << "Tuples size = " << tuples.size() << endl;
	_file.close();

	if (tuples.size() == 0)
	{
		cout << "data is NULL." << endl;
		return -2;
	}

	//数据归一化
	int i, j;
	//使用全局最大值最小值进行归一化
	if (is_normalization == 1)
	{
		cout << "start normalization..." << endl;
		double max_val = tuples[0][1], min_val = tuples[0][1];
		for (i = 0; i < tuples.size(); i++)
		{
			for (j = 1; j < nFeaturesCount + 1; j++)
			{
				max_val = tuples[i][j] > max_val ? tuples[i][j] : max_val;
				min_val = tuples[i][j] < min_val ? tuples[i][j] : min_val;
			}
		}

		for (i = 0; i < tuples.size(); i++)
		{
			for (j = 1; j < nFeaturesCount + 1; j++)
			{
				tuples[i][j] = (tuples[i][j] - min_val) / (max_val - min_val);
			}
			
		}

		cout << "Normalization success." << endl;
	}
	
	//对每个特征进行归一化
	if (is_normalization == 2)
	{
		cout << "start normalization..." << endl;
		
		//double max_val = tuples[0][1], min_val = tuples[0][1];
		double* pd_max_val = new double[nFeaturesCount];
		double* pd_min_val = new double[nFeaturesCount];
		
		for (int i = 0; i < nFeaturesCount; i++)
		{
			pd_max_val[i] = tuples[0][i + 1];
			pd_min_val[i] = tuples[0][i + 1];
		}

		for (i = 0; i < tuples.size(); i++)
		{
			for (j = 1; j < nFeaturesCount + 1; j++)
			{
				pd_max_val[j - 1] = tuples[i][j] > pd_max_val[j - 1] ? tuples[i][j] : pd_max_val[j - 1];
				pd_min_val[j - 1] = tuples[i][j] < pd_min_val[j - 1] ? tuples[i][j] : pd_min_val[j - 1];
			}
		}

		for (i = 0; i < tuples.size(); i++)
		{
			for (j = 1; j < nFeaturesCount + 1; j++)
			{
				tuples[i][j] = (tuples[i][j] - pd_min_val[j - 1]) / (pd_max_val[j - 1] - pd_min_val[j - 1]);
			}
		}

		delete[] pd_max_val;
		delete[] pd_min_val;
		cout << "Normalization success." << endl;
	}

	cout << "programmed by Yaoyao & Maike & Kampau. Copyright 2016." << endl;
	cout << "any problem please contact: whuyao@foxmail.com" << endl;

	//开始聚类处理
	if (ndisttype < 0)
	{
		vector<Tuple>* clusters;
		DTW_KMeans(tuples, cluster_num, nMaxIterations, dMinError, ndisttype, clusters, isUseRand, max_gap_dist, nClusterNumTimes, nRandParts, means_filename.toStdString().data(), dMininInitContribution, distance_filename.toStdString().data());
		outputFile(clusters, cluster_num, outputfilename);

		float* pSilVals = NULL;
		float aveSilVal = 0;
		silhouette_output(tuples, clusters, cluster_num, ndisttype, pSilVals, aveSilVal, eva_filename.toStdString().data(), distance_filename.toStdString().data());

		delete[]clusters;
	}
	else
	{
		real_2d_array _tdata;
		real_1d_array _dataids;
		_tdata.setlength(tuples.size(), nFeaturesCount);
		_dataids.setlength(tuples.size());

		for (i = 0; i < tuples.size(); i++)
		{
			_dataids[i] = tuples[i][0];
			for (j = 1; j < nFeaturesCount + 1; j++)
			{
				_tdata[i][j - 1] = tuples[i][j];
			}
		}

		TRAN_KMeans(_tdata, _dataids, outputfilename, cluster_num, ndisttype);

	}

	cout << "******************** KMEANS ********************" << endl;

	cout << "programmed by Yaoyao & Maike & Kampau. Copyright 2016." << endl;
	cout << "any problem please contact: whuyao@foxmail.com" << endl;

	return 0;
}
