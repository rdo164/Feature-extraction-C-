// hue_histogram.hpp
#ifndef HUEHIST_H
#define HUEHIST_H
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>
#include <string>
#include "hue_histogram.hpp"

using namespace cv;
using namespace std;

// Build a normalized hue histogram (180 bins) from a BGR image.
// - s_threshold / v_threshold: reject low-saturation / low-value pixels
// - center_hue in [0,179]: rotate hue wheel so this hue is centered; -1 disables
// - roiMask: optional mask to limit histogram to an object (8-bit, 0 or 255)
inline cv::Mat createHueHistogram(const cv::Mat& bgr,
                                  int s_threshold = 30,
                                  int v_threshold = 30,
                                  int center_hue   = -1,
                                  const cv::Mat& roiMask = cv::Mat())
{
    CV_Assert(!bgr.empty() && bgr.type() == CV_8UC3);

    // 1) BGR -> HSV, split
    cv::Mat hsv; cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch; cv::split(hsv, ch);
    cv::Mat H = ch[0], S = ch[1], V = ch[2];

    // 2) Optional hue re-centering via LUT (fast, no modulo hacks)
    if (0 <= center_hue && center_hue <= 179) {
        cv::Mat lut(1, 256, CV_8U);
        for (int i = 0; i < 256; ++i) {
            // Only 0..179 are valid hue values; others (180..255) are unused but mapped safely.
            int raw = i;
            if (raw > 179) raw = 179;
            int shifted = (raw - center_hue + 90) % 180;
            if (shifted < 0) shifted += 180;
            lut.at<uchar>(0, i) = static_cast<uchar>(shifted);
        }
        cv::LUT(H, lut, H);
    }

    // 3) Mask for meaningful colors: S >= s_threshold AND V >= v_threshold
    cv::Mat sMask, vMask, colorMask;
    cv::inRange(S, cv::Scalar(s_threshold), cv::Scalar(255), sMask);
    cv::inRange(V, cv::Scalar(v_threshold), cv::Scalar(255), vMask);
    cv::bitwise_and(sMask, vMask, colorMask);

    // 4) If a ROI mask for the current object exists, combine with colorMask
    cv::Mat finalMask = colorMask;
    if (!roiMask.empty()) {
        CV_Assert(roiMask.type() == CV_8U && roiMask.size() == bgr.size());
        cv::bitwise_and(colorMask, roiMask, finalMask);
    }

    // 5) Histogram on H with masking
    int histSize = 180;
    float range[] = {0.f, 180.f};
    const float* histRange = range;
    cv::Mat hist; // CV_32F, 180x1
    int channels[] = {0};
    cv::calcHist(&H, 1, channels, finalMask, hist, 1, &histSize, &histRange, true, false);

    // 6) Normalize to [0,1] for comparability
    if (!hist.empty())
        cv::normalize(hist, hist, 0.0, 1.0, cv::NORM_MINMAX);

    return hist; // 180x1, CV_32F
}

struct config {
    Mat img;
    float sensitivity;
    int window;
    int desiredHue;
    bool exclude;
};

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
static Mat extract_features(const Mat& src, const Mat& objMask, const Mat& result)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // findContours modifies the image; so I use clone
    findContours(objMask.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    cout << "Number of contours (all): " << contours.size() << endl;

    int piezas = 0; // counter for large objects

    Mat obj_pixels;
    Mat mask = Mat::zeros(src.size(), CV_8UC1);
    for (size_t i = 0; i < contours.size(); ++i) 
    {
        double area = contourArea(contours[i]);
        if (area > 20000) { // if not it does not recognize properly
            piezas++;

            // mask of object i
            drawContours(mask, contours, (int)i, Scalar(255), FILLED);

            // cropped colored object
            result.copyTo(obj_pixels, mask);
        }
    }
    //create_window("masked", obj_pixels, 500, 300, 700 ,0);
    // Draw all contours on a copy of the original image

   
    // for (size_t i = 0; i < piezas_imgs.size(); ++i) {

    //     // ventana para la pieza i
    //     std::string winPiece  = "Pieza " + std::to_string(i);
    //     // ventana para el histograma de la pieza i
    //     std::string winHist   = "Hue Histograma " + std::to_string(i);

    //     create_window(winPiece, piezas_imgs[i], 500, 300, 50,  50 + 430 * i);

    //     // 1) Vector hue-frecuencia (180 valores 0..1)
    //     Mat hist = createHueHistogram(piezas_imgs[i]);

    //     std::cout << hist;

    //     // 2) Imagen 2D del histograma
    //     Mat histImg = drawHueHistogram(hist);

    //     create_window(winHist, histImg, 500, 300, 700, 50 + 430 * i);
    // }

    //     // 1) Vector hue-frecuencia (180 valores 0..1)
    Mat hist = createHueHistogram(obj_pixels);
    std::cout << hist;

    
        // 2) Imagen 2D del histograma
    // Mat histImg = drawHueHistogram(hist);

    // create_window("full Picture", histImg, 500, 300, 700, 50 + 430);
    return hist;
}

bool to_push(const config & cfg, Mat & hist) {
    float totalDesired = 0;
    float total = 0;

    int upper = cfg.desiredHue + cfg.window;
    int lower = cfg.desiredHue - cfg.window;

    std::cout << "\n\nupper: " << upper << " lower: " << lower;

    for(int i = 0; i < hist.rows; i++)
    {
        total += hist.at<float>(i,0);
        if (i >= lower && i <= upper) {
            std::cout << "\nAdding: " << hist.at<float>(i,0) << " At: " << i;
            totalDesired += hist.at<float>(i,0);
        }

        else if (lower < 0 && i >= (hist.rows + lower - 1)) {
            std::cout << "\nAdding: " << hist.at<float>(i,0) << " At: " << i;
            totalDesired += hist.at<float>(i,0);
        }

        else if (upper >= hist.rows && i <= (upper - hist.rows - 1)) {
            std::cout << "\nAdding: " << hist.at<float>(i,0) << " At: " << i;
            totalDesired += hist.at<float>(i,0);
        }
    }

    std::cout << "\n\nTotal: " << total;
    std::cout << "\n\nTotalDesired: " << totalDesired;

    if (cfg.exclude) return total - totalDesired > total * (1-cfg.sensitivity);
    else return totalDesired > total * (1-cfg.sensitivity);
}


// PIPELINE
bool run_pipeline(const config & cfg)
{

    // (2) Enhancement
    HSVData hsvd;
    enhance_image(cfg.img, hsvd);  

    // (3) Segmentation
    Mat objMask, foreground;
    segment_image(cfg.img, hsvd, objMask, foreground);

    // Visualization of intermediate results
    //create_window("Original",   cfg.img,       500, 300,   0,   0);
    //create_window("Mask",       objMask,   500, 300, 100,   0);
    //create_window("Foreground", foreground,500, 300, 300,   0);

    // (4) Feature Extraction  
    Mat hist = extract_features(cfg.img, objMask, foreground);

    return to_push(cfg, hist);
}
#endif