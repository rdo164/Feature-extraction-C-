#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>
#include <string>
#include "hue_histogram.hpp"

using namespace cv;
using namespace std;

// ============================================================================
// Function to create and show a window at specific position and size
static void create_window(const string& name, const Mat& img,
                          int w, int h, int x, int y)
{
    namedWindow(name, WINDOW_NORMAL);     // allows resizing
    resizeWindow(name, w, h);             // set desired size
    moveWindow(name, x, y);               // set position
    imshow(name, img);                    // display content
}

// 1) IMAGE ACQUISITION
static bool acquire_image(const string& source, Mat& src_bgr)
{
    src_bgr = imread(source, IMREAD_COLOR);
    if (src_bgr.empty()) { cerr << "[ACQ] Could not read: " << source << "\n"; return false; }
    return true;
}

// 2) IMAGE ENHANCEMENT  (HSV split + GaussianBlur )
struct HSVData {
    Mat H, S, V;      // originales
    Mat V_smooth;     // V suavizada para segmentar
};

static void enhance_image(const Mat& src_bgr, HSVData& hsvd)
{
    Mat hsv;
    cvtColor(src_bgr, hsv, COLOR_BGR2HSV); // conversion to HSV
    vector<Mat> ch; split(hsv, ch);// split channels
    // store copies of H, S, V
    hsvd.H = ch[0].clone(); 
    hsvd.S = ch[1].clone();
    hsvd.V = ch[2].clone();
    // gaussian blur on V channel for smoothing
    GaussianBlur(hsvd.V, hsvd.V_smooth, Size(3,3), 0.0); 
}

// ============================================================================
// 3) SEGMENTATION  (Otsu on V_smooth + filter on  S + morphology + foreground)
static void segment_image(const Mat& src_bgr, const HSVData& hsvd,
                          Mat& objMask, Mat& foreground)
{
    // 3.1) Otsu threshold on V channel (for bright objects)
    threshold(hsvd.V_smooth, objMask, 0, 255, THRESH_BINARY | THRESH_OTSU); 

    // 3.2) Keep only pixels with medium/high saturation
    Mat satKeep;
    inRange(hsvd.S, Scalar(20), Scalar(255), satKeep); // keep pixels with S > 20 
    bitwise_and(objMask, satKeep, objMask); // combine masks

    // 3.3) Morphological cleaning
    Mat k_open  = getStructuringElement(MORPH_ELLIPSE, Size(9,9));// OPEN: removes small white noise
    Mat k_close = getStructuringElement(MORPH_ELLIPSE, Size(7,7));// close: fill gaps

    morphologyEx(objMask, objMask, MORPH_OPEN,  k_open);// removes small white noise
    morphologyEx(objMask, objMask, MORPH_CLOSE, k_close);// fills small black holes inside white objects

    // 3.4) Apply mask to original image (colored foreground)
    bitwise_and(src_bgr, src_bgr, foreground, objMask);
}

// function to get approximate color name from BGR mean color
struct NamedColor {
    string name;
    Scalar bgr; // formato BGR
};
// Function that returns a color name based on HSV values
string getColorNameFromHSV(const Scalar& hsv) {
    float h = hsv[0]; // Hue [0,180] en OpenCV
    float s = hsv[1]; // Saturation [0,255]
    float v = hsv[2]; // Value [0,255]

    if (v < 50)
        return "Black";
    if (s < 50 && v > 200)
        return "white";
    if (s < 50)
        return "Grey";

    // Clasificación por Hue (valores típicos OpenCV)
    if (h < 10 || h >= 170)
        return "Red";
    else if (h >= 10 && h < 25)
        return "Orange";
    else if (h >= 25 && h < 35)
        return "Yellow";
    else if (h >= 35 && h < 85)
        return "Green";
    else if (h >= 85 && h < 125)
        return "Cian";
    else if (h >= 125 && h < 145)
        return "Azul";
    else if (h >= 145 && h < 170)
        return "Magenta";

    return "Unknown";
}

// ============================================================================
// 4) FEATURE EXTRACTION  (Contours + HSV mean
static void extract_features(const Mat& src, const Mat& objMask, const Mat& result)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // findContours modifies the image; so I use clone
    findContours(objMask.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    cout << "Number of contours (all): " << contours.size() << endl;

    int piezas = 0; // counter for large objects
    vector<Mat> piezas_imgs;

    for (size_t i = 0; i < contours.size(); ++i) 
    {
        double area = contourArea(contours[i]);
        if (area > 20000) { // if not it does not recognize properly
            piezas++;

            // mask of object i
            Mat mask = Mat::zeros(result.size(), CV_8UC1);
            drawContours(mask, contours, (int)i, Scalar(255), FILLED);

            // cropped colored object
            Mat obj_pixels;
            result.copyTo(obj_pixels, mask);
            piezas_imgs.push_back(obj_pixels);

            // mean BGR color
            Scalar mean_color = mean(result, mask);
            // obtener el nombre aproximado del color
            string colorName = getColorNameFromHSV(mean_color);

           cout << "Object " << i
             << " - area: " << area 
             << " - mean color (BGR): (" << mean_color[0]
             << ", " << mean_color[1]
             << ", " << mean_color[2] << ")"
             << " - Color aproximado: " << colorName << endl;
               }
    }
    // Draw all contours on a copy of the original image
    Mat vis = src.clone();
    drawContours(vis, contours, -1, Scalar(0,255,0), 2);
    create_window("Contours", vis, 500, 300, 700, 0);

   
    for (size_t i = 0; i < piezas_imgs.size(); ++i) {
        Mat hsv;
        cvtColor(piezas_imgs[i], hsv, COLOR_BGR2HSV);

        // ---- calcHist (H channel) ----
        int h_bins = 30, sbins= 32;
        int histSize[] = { h_bins, sbins };
        float h_ranges[] = { 0, 180 };
        float s_ranges[] = { 0, 255 };

        const float* ranges[] = { h_ranges, s_ranges };
        int channels[] = { 0,1 };


        cv::Mat hist;
        cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
        // validate histogram values
        for (int i = 0; i < hist.rows; i++) {
           cout << "Bin " << i << ": " << hist.at<float>(i) << endl;
        }

        // create histogram image
        double maxVal=0;

        minMaxLoc(hist, 0, &maxVal, 0, 0);
        
        int scale = 10;
        
        Mat histImg = Mat::zeros(sbins, h_bins, CV_8UC3);

        for( int h = 0; h < h_bins; h++ )
        {
            for( int s = 0; s < sbins; s++ )
            {
                float binVal = hist.at<float>(h, s);

                int intensity = cvRound(binVal*255/maxVal);

                rectangle( histImg, Point(h*scale, s*scale),
                            Point( (h+1)*scale - 1, (s+1)*scale - 1),
                            Scalar::all(intensity),
                            -1 );
            }
        }
        namedWindow( "H-S Histogram", 1 );
        create_window( "H-S Histogram", histImg, 512, 400, 1200, 0 );
        imshow("H-S Histogram " + to_string(i), histImg);

        // int hist_w = 512, hist_h = 400;
        // int bin_w  = cvRound((double)hist_w / h_bins);
        // cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

        // cv::normalize(hist, hist, 0, histImage.rows - 1, cv::NORM_MINMAX);

        // for (int j = 1; j < h_bins; ++j) {
        //     float h0 = hist.at<float>(j - 1);
        //     float h1 = hist.at<float>(j);

        //     int x0 = bin_w * (j - 1);
        //     int x1 = bin_w * j;
        //     int y0 = hist_h - cvRound(h0);
        //     int y1 = hist_h - cvRound(h1);

        //     cv::line(histImage,
        //             cv::Point(x0, y0),
        //             cv::Point(x1, y1),
        //             cv::Scalar(0, 0, 255),
        //             2,
        //             cv::LINE_AA);
        // }
        // create_window("object ",piezas_imgs[i],600, 400, 0, 0);
        // cv::imshow("Object " + std::to_string(i), hsv);
        // cv::imshow("Hue Histogram " + std::to_string(i), histImage);
    
    }
    waitKey(0);

}


// PIPELINE
int run_pipeline(const string& imgPath)
{
    // (1) Acquisition
    Mat src;
    if (!acquire_image(imgPath, src)) return -1;

    // (2) Enhancement
    HSVData hsvd;
    enhance_image(src, hsvd);  

    // (3) Segmentation
    Mat objMask, foreground;
    segment_image(src, hsvd, objMask, foreground);

    // Visualization of intermediate results
    create_window("Original",   src,       500, 300,   0,   0);
    create_window("Mask",       objMask,   500, 300, 100,   0);
    create_window("Foreground", foreground,500, 300, 300,   0);

    // (4) Feature Extraction  
    extract_features(src, objMask, foreground);

    waitKey(0);
    return 0;
}


// ============================================================================
// main
// ============================================================================
int main()
{
    Mat img = imread("../img/IMG_8819.JPG");
    
    return run_pipeline("../img/IMG_8819.JPG");

    waitKey(0);

}
