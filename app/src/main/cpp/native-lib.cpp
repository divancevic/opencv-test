#include <jni.h>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define TAG "NativeLib"

using namespace std;
using namespace cv;

void resize(Mat & image, int resize) {
    cv::Size img = image.size();

    float height_ = img.height;
    float width_ = img.width;

    float scale = (float)resize/(float)height_;
    width_ = (int) (width_ * scale);

    cv::resize(image, image, cv::Size(width_, resize), INTER_AREA);
}

bool compareContoursAreas(vector<Point> contour1, vector<Point> contour2) {
	double i = fabs(contourArea(Mat(contour1)));
	double j = fabs(contourArea(Mat(contour2)));
	return (i > j);
}



extern "C" {
jboolean JNICALL
Java_com_example_nativeopencvandroidtemplate_MainActivity_adaptiveThresholdFromJNI(JNIEnv *env,
                                                                                   jobject instance,
                                                                                   jlong matAddr,
                                                                                   jlong roi) {

    // Get Mat from raw address
    Mat &image = *(Mat *) matAddr;
    Mat &roi_ = *(Mat *) roi;

    cv::Size _shape_ = image.size();
    if(_shape_.height  == 0 || _shape_.width == 0) {
        return false;
    }

    // Show central rectangle
    Point x1(_shape_.width * 0.02, _shape_.height * 0.02);
    Point x2(_shape_.width * 0.98, _shape_.height * 0.98);

    rectangle(image, x1, x2, Scalar(0,255,0), 2, LINE_8);

    Mat gray;
    cv::cvtColor(image, gray, COLOR_RGBA2GRAY);

    // Set kernels

    const Mat rectKernel = cv::getStructuringElement(MORPH_RECT, cv::Size(5,13));
    const Mat sqKernel = cv::getStructuringElement(MORPH_RECT, cv::Size(21,21));
    const Mat square = cv::getStructuringElement(MORPH_RECT, cv::Size(2,2));

    Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(3,3), 0);

    Mat blackhat;
    cv::morphologyEx(blur, blackhat, MORPH_BLACKHAT, rectKernel);

    Mat gradX;
    cv::Sobel(blackhat, gradX, CV_32F, 1, 0, -1, 1, 0 , BORDER_DEFAULT);

    cv::abs(gradX);

    double min_, max_;
    cv::minMaxIdx(gradX, &min_, &max_);

    cv::Size shape_ = gradX.size();

    Mat gradXAbs(shape_.height, shape_.width, CV_8UC1);

    for(int i=0; i < shape_.height; i++) {
        for(int j=0; j < shape_.width; j++) {
        gradXAbs.at<unsigned char>(i,j) = (unsigned char)(255.0 * ((gradX.at<float>(i,j) - min_) / (max_ - min_)));
        }
    }

    Mat closingOperation;
    cv::morphologyEx(gradXAbs, closingOperation, MORPH_CLOSE, rectKernel);

    Mat threshold_;
    threshold(closingOperation, threshold_, 0, 255, THRESH_OTSU | THRESH_BINARY);

    cv::morphologyEx(threshold_, closingOperation, MORPH_CLOSE, sqKernel);

    Mat eroded = closingOperation;

    cv::Size erode_size = eroded.size();
    int p = 0.03 * erode_size.width;

    // left
    for(int i=0; i < erode_size.height; i++) {
        for(int j=0; j < p; j++) {
            eroded.at<unsigned char>(i,j) = 0;
         }
    }

    // right
    for(int i=0; i < erode_size.height; i++) {
        for(int j=erode_size.width - p; j < erode_size.width; j++) {
            eroded.at<unsigned char>(i,j) = 0;
        }
    }

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(eroded, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(), compareContoursAreas);

    for(size_t i = 0; i < contours.size(); i++) {
        Rect rect = boundingRect(contours[i]);
        float ar = rect.width / float(rect.height);

        float crWidth =  float(rect.width)/float(shape_.width);

        if(ar > 5.0 && crWidth > 0.75) {
            int x = int(rect.x);
            int y = int(rect.y);
            int pX = (int)((rect.x + rect.width) * 0.03);
            int pY = (int)((rect.y + rect.height) * 0.03);

            x = x - pX;
            y = y - pY;

            int w = rect.width + (pX * 2);
            int h = rect.height + (pY * 2);


            if( (x + w > shape_.width) ||
            (y + h > shape_.height) ) {
                continue;
            }

            roi_ = image(Rect(x, y, w, h));
            cv::cvtColor(roi_, roi_, COLOR_RGBA2GRAY);

            rectangle(image, Point(x, y), Point(x + w,  y + h ), Scalar(230,230,250), 2, LINE_8);

            return true;
        }
    }
    return false;
}
}