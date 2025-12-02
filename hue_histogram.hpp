// hue_histogram.hpp
#pragma once
#include <opencv2/opencv.hpp>

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
