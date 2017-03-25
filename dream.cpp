// this is a cpp translation of deepdream python code
// some functions are copied from caffe cpp classification and stackoverflow
// the last 2 layers have to be removed from googlenet prototxt to run
// why?? see: http://stackoverflow.com/questions/42634179/caffenet-reshape
// But in python implementation that is not needed. why??
// I looked into _caffe.cpp and pycaffe.py, but found nothing
// I tested using both openblas and mkl and numpy functions are quite faster (mkl) than opencv
// any suggestion is welcome at momen_bhuiyan@yahoo.com
// edit summary: I previously thought there was a bottleneck in forward-backward due to time computation using clock() as it calculated clock cycle used by all processor


#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ctime>

// from stackoverflow
#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

// const
const int num_channels_=3;
const string end = "inception_3b/output";
const string end2 = "inception_4c/output";

// vars
shared_ptr<Net<float> > net_;
cv::Mat img,guide;
float* guide_features;
float* guide_features_t;
int guide_features_area;
int index_of_end;
std::vector<cv::Mat> input_channels;

// cvtype to string from stackoverflow
string type2str(int type) {
  string r;
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }
  r += "C";
  r += (chans+'0');
  return r;
}

// wrap layer pointers
void WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	input_channels->clear();
	Blob<float>* input_layer = net_->input_blobs()[0];
	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

// opencv image pointers to wrapped layer pointers
void cvimg_mat_to_blob(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
	cv::split(img, *input_channels);
	CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())<< "Input channels are not wrapping the input layer of the network.";
}

// add sub ilsvrc mean
cv::Mat sub_mean(const cv::Mat& img){
	cv::Mat sample_normalized;
	cv::Scalar channel_mean(104.0, 116.0, 122.0);
  	cv::Mat mean_mat =cv::Mat(cv::Size(img.cols,img.rows), img.type(), channel_mean);
	cv::subtract(img,mean_mat, sample_normalized);
	return sample_normalized;
}


cv::Mat add_mean(const cv::Mat& img){
	cv::Mat sample_normalized;
	cv::Scalar channel_mean(104.0, 116.0, 122.0);
  	cv::Mat mean_mat =cv::Mat(cv::Size(img.cols,img.rows), img.type(), channel_mean);
	cv::add(img,mean_mat, sample_normalized);
	return sample_normalized;
}


void display_image(const cv::Mat& image){
	cv::Mat tmp = add_mean(image),dst;
	cv::normalize(tmp, dst, 0, 1, cv::NORM_MINMAX);
	cv::namedWindow("Display window");
    cv::imshow("Display window", dst);
    cv::waitKey(0);
}


void save_imgmat_in_file(const cv::Mat& image,const char * filename){
	cv::Mat tmp = add_mean(image);
	cv::imwrite(filename,tmp);
}


cv::Mat blob_to_cvimg_mat(const cv::Size ss,float * data){
	std::vector<cv::Mat> channels;
	for (int i = 0; i < 3; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(ss, CV_32FC1, data);
		channels.push_back(channel);
		data += ss.area();
	}
	cv::Mat res;
	cv::merge(channels, res);
	return res;
}


// out should have same size
// copied from stackoverflow
void shiftCol(cv::Mat& out, cv::Mat in, int numRight){
    if(numRight == 0){ 
        in.copyTo(out);
        return;
    }
    int ncols = in.cols;
    int nrows = in.rows;
    // out = cv::Mat::zeros(in.size(), in.type());
    numRight = numRight%ncols;
    if(numRight < 0)numRight = ncols+numRight;
    in(cv::Rect(ncols-numRight,0, numRight,nrows)).copyTo(out(cv::Rect(0,0,numRight,nrows)));
    in(cv::Rect(0,0, ncols-numRight,nrows)).copyTo(out(cv::Rect(numRight,0,ncols-numRight,nrows)));
}

void shiftRows(cv::Mat& out, cv::Mat in, int numpos){
    if(numpos == 0){ 
        in.copyTo(out);
        return;
    }
    int ncols = in.cols;
    int nrows = in.rows;
    // out = cv::Mat::zeros(in.size(), in.type());
    numpos = numpos%nrows;
    if(numpos < 0)numpos = nrows+numpos;
    in(cv::Rect(0,nrows-numpos, ncols,numpos)).copyTo(out(cv::Rect(0,0,ncols,numpos)));
    in(cv::Rect(0,0, ncols,nrows-numpos)).copyTo(out(cv::Rect(0,numpos,ncols,nrows-numpos)));
}


void objective_L2(){
	memcpy(net_->blob_by_name(end).get()[0].mutable_cpu_diff(),net_->blob_by_name(end).get()[0].mutable_cpu_data(),net_->blob_by_name(end)->channels()*net_->blob_by_name(end)->height()*net_->blob_by_name(end)->width()*sizeof(float));
}


//objective for guided feature
void objective(){
	int net_area = net_->blob_by_name(end)->width()*net_->blob_by_name(end)->height();
	int net_chan = net_->blob_by_name(end)->channels();
	std::vector<int> idxs;
	cv::Point p2;

	clock_t begint = clock();
	float *res = new float[guide_features_area*net_area];
	// this is ~10X slower if compiled with openblas as numpy mkl dot is way fast
	caffe_cpu_gemm<float>(CblasNoTrans,CblasNoTrans,guide_features_area,net_area,net_chan,1,guide_features_t,net_->blob_by_name(end)->mutable_cpu_data(),0,res);
	cv::Mat rr = cv::Mat(cv::Size(net_area,guide_features_area), CV_32FC1, res);
	clock_t endt = clock();
	double time_spent = (double)(endt - begint) / CLOCKS_PER_SEC;
	printf("timespent in cross : %lf \n",time_spent );
	// argmaxs
	for (int i = 0; i < net_area; ++i)
	{
		cv::minMaxLoc(rr.col(i),NULL,NULL,NULL,&p2);
		idxs.push_back(p2.y);
	}
	// todo: parallelize copy
	float * data =guide_features;
	float * diff =net_->blob_by_name(end)->mutable_cpu_diff();
	for (int i = 0; i < net_chan; ++i)
	{
		for (int j = 0; j < net_area; ++j)
		{
			diff[j] = data[idxs[j]];
		}
		data += guide_features_area;
		diff += net_area;
	}
	delete[] res;
}


// copied from stackoverflow
void clamp(cv::Mat& mat, cv::Point3f lowerBound, cv::Point3f upperBound) {
    vector<cv::Mat> matc;
    split(mat, matc);
    min(max(matc[0], lowerBound.x), upperBound.x, matc[0]);
    min(max(matc[1], lowerBound.y), upperBound.y, matc[1]);
    min(max(matc[2], lowerBound.z), upperBound.z, matc[2]);
    merge(matc, mat);   
}


// todo: change rand to caffe rng
void make_step(cv::Size octave_base_size,float step_size=1.5, int jitter=32,bool clip=true){
	cv::Mat src = blob_to_cvimg_mat(octave_base_size,net_->input_blobs()[0]->mutable_cpu_data());
	int ox = (rand()%(jitter*2+1))-jitter;
	int oy = (rand()%(jitter*2+1))-jitter;
	cv::Mat res = cv::Mat::zeros(src.size(), src.type());
	shiftRows(res,src,ox);
	shiftCol(src,res,oy);

	// cv::merge creates new array
	cvimg_mat_to_blob(src,&input_channels);
	clock_t begint = clock();
	net_->ForwardFromTo(0,index_of_end);
	clock_t endt = clock();
	double time_spent = (double)(endt - begint) / CLOCKS_PER_SEC;
	printf("timespent in forward : %lf \n",time_spent );
	begint = endt;
	objective();
	endt = clock();
	time_spent = (double)(endt - begint) / CLOCKS_PER_SEC;
	printf("timespent in objective : %lf \n",time_spent );
	begint = endt;
	net_->BackwardFromTo(index_of_end,0);
	endt = clock();
	time_spent = (double)(endt - begint) / CLOCKS_PER_SEC;
	printf("timespent in backward : %lf \n",time_spent );
	
	// update image
	float asum_mean =  caffe_cpu_asum<float>(octave_base_size.area() * num_channels_, net_->input_blobs()[0]->cpu_diff())/(octave_base_size.area() * num_channels_);
	float stepp = step_size/asum_mean;
	caffe_axpy<float>(octave_base_size.area() * num_channels_, stepp, net_->input_blobs()[0]->cpu_diff(),net_->input_blobs()[0]->mutable_cpu_data());
	src = blob_to_cvimg_mat(octave_base_size,net_->input_blobs()[0]->mutable_cpu_data());
	
	// reverse roll
	shiftRows(res,src,-ox);
	shiftCol(src,res,-oy);

	if(clip){
		clamp(src,cv::Point3f(-104.0, -116.0, -122.0),cv::Point3f(255-104.0, 255-116.0, 255-122.0));
	}
	cvimg_mat_to_blob(src,&input_channels);
}



void dream(int times=4,float scale=1.4,int iter_n=10){
	// create resized images
	std::vector<cv::Mat> octaves;
	octaves.push_back(img);
	for (int i = 1; i < times; ++i)
	{
		cv::Mat tmp;
		cv::resize(octaves[i-1], tmp, cv::Size( octaves[i-1].cols/scale, octaves[i-1].rows/scale ) );
		octaves.push_back(tmp);
	}
	cv::Mat detail = cv::Mat::zeros(octaves[times-1].size(),octaves[times-1].type());
	for (int octave = times-1; octave >= 0; octave--)
	{
		cv::Mat octave_base = octaves[octave];
		if(octave != times-1){
			cv::Mat tmp;
            cv::resize(detail, tmp,octave_base.size());
            detail = tmp;
		}
		
		// reshape input
		cv::Mat ttt = octave_base+detail;
		Blob<float>* input_layer = net_->input_blobs()[0];
  		input_layer->Reshape(1, num_channels_,ttt.rows, ttt.cols);
		net_->Reshape();
		WrapInputLayer(&input_channels);
		cvimg_mat_to_blob(ttt, &input_channels);

		string filename = "octave"+SSTR(octave)+".jpg";
		string filename2 = "display"+SSTR(octave)+".jpg";
		string filename3 = "result"+SSTR(octave)+".jpg";
		clock_t begint = clock();
		for (int i = 0; i < iter_n; ++i)
		{
			make_step(octave_base.size());
			
			// cv::Mat image =blob_to_cvimg_mat(octave_base.size(),net_->input_blobs()[0]->mutable_cpu_data());
			// display_image(image);
		}
		clock_t endt = clock();
		double time_spent = (double)(endt - begint) / CLOCKS_PER_SEC;
		printf("timespent on iteration %d : %lf \n",octave,time_spent );
		detail = blob_to_cvimg_mat(octave_base.size(),net_->input_blobs()[0]->mutable_cpu_data())-octave_base;
		cv::Mat image =blob_to_cvimg_mat(octave_base.size(),net_->input_blobs()[0]->mutable_cpu_data());
		save_imgmat_in_file(detail,filename2.c_str());
		save_imgmat_in_file(image,filename3.c_str());

	}
}


int main(int argc, char const *argv[])
{
	if (argc != 5) {
	    std::cerr << "Usage: " << argv[0]
	              << " deploy.prototxt network.caffemodel"
	              << " sky.jpg flower.jpg" << std::endl;
	    return 1;
	}
	srand(time(NULL));
	// argv
	string model_file   = argv[1];
	string trained_file = argv[2];
	string sky = argv[3];
	string flower = argv[4];
	Caffe::set_mode(Caffe::CPU);
		
	// load model and param
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);
	
	// get index of
	const vector<string>&lays =net_->layer_names();
	for (int i = 0; i < lays.size(); ++i)
	{
		if(end == lays[i])index_of_end = i;
	}

	// load image and preprocess
	cv::Mat img_main = cv::imread(sky, -1);
  	cv::Mat guide_main = cv::imread(flower, -1);
  	cv::Mat img_tmp,guide_tmp;
  	img_main.convertTo(img_tmp, CV_32FC3);
  	guide_main.convertTo(guide_tmp, CV_32FC3);
  	img = sub_mean(img_tmp);
  	guide = sub_mean(guide_tmp);

  	// reshape net
  	Blob<float>* input_layer = net_->input_blobs()[0];
  	input_layer->Reshape(1, num_channels_,guide.rows, guide.cols);
	net_->Reshape();

	// forward the guide
	WrapInputLayer(&input_channels);
	cvimg_mat_to_blob(guide, &input_channels);
	net_->ForwardFromTo(0,index_of_end);
	
	//copy to guide_features
	guide_features_area = net_->blob_by_name(end)->width()*net_->blob_by_name(end)->height();
	int guide_chan = net_->blob_by_name(end)->channels();

	int guide_size = sizeof(float)*guide_chan*guide_features_area;
	guide_features = new float[guide_chan*guide_features_area];
	memcpy(guide_features,net_->blob_by_name(end)->mutable_cpu_data(),guide_size);

	// transpose
	guide_features_t = new float[guide_chan*guide_features_area];
	float *tmp = guide_features_t;
	for (int j = 0; j < guide_features_area; ++j)
	{
		for (int i = 0; i < guide_chan; ++i)
		{
			tmp[0] = guide_features[i*guide_features_area+j];
			tmp++;
		}
		
	}
	
	// dream
	dream();
	delete[] guide_features;
	delete[] guide_features_t;
	return 0;
}