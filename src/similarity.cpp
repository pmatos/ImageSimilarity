#include <bitset>
#include <iostream>
#include <opencv2/opencv.hpp>

uint64_t computeDHash(const cv::Mat &img) {
  cv::Mat resizedImg, grayImg;
  cv::resize(img, resizedImg, cv::Size(9, 8));
  cv::cvtColor(resizedImg, grayImg, cv::COLOR_BGR2GRAY);

  uint64_t hash = 0;
  for (int row = 0; row < grayImg.rows; ++row) {
    for (int col = 0; col < grayImg.cols - 1; ++col) {
      if (grayImg.at<uint8_t>(row, col) > grayImg.at<uint8_t>(row, col + 1)) {
        hash |= 1ULL << (row * 8 + col);
      }
    }
  }
  return hash;
}

int hammingDistance(uint64_t hash1, uint64_t hash2) {
  return std::bitset<64>(hash1 ^ hash2).count();
}

double compareImagesHashed(const cv::Mat &img1, const cv::Mat &img2) {
  if (img1.empty() || img2.empty()) {
    std::cerr << "Error: Empty input image(s)" << std::endl;
    return 0.0;
  }

  uint64_t hash1 = computeDHash(img1);
  uint64_t hash2 = computeDHash(img2);
  int distance = hammingDistance(hash1, hash2);

  // Normalize the Hamming distance to a similarity score in the range [0, 1].
  return 1.0 - static_cast<double>(distance) / 64.0;
}

double compareImages(const cv::Mat &img1, const cv::Mat &img2) {
  if (img1.empty() || img2.empty()) {
    std::cerr << "Error: Empty input image(s)" << std::endl;
    return 0.0;
  }

  cv::Mat resizedImg1, resizedImg2, grayImg1, grayImg2, diff;

  cv::resize(img1, resizedImg1, cv::Size(256, 256));
  cv::resize(img2, resizedImg2, cv::Size(256, 256));

  cv::cvtColor(resizedImg1, grayImg1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(resizedImg2, grayImg2, cv::COLOR_BGR2GRAY);

  cv::absdiff(grayImg1, grayImg2, diff);
  int totalPixels = diff.rows * diff.cols;
  int matchingPixels = totalPixels - cv::countNonZero(diff);

  return static_cast<double>(matchingPixels) / totalPixels;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: image_similarity img1_path img2_path" << std::endl;
    return 1;
  }

  cv::Mat img1 = cv::imread(argv[1]);
  cv::Mat img2 = cv::imread(argv[2]);

  double similarity = compareImages(img1, img2);
  double similarityHashed = compareImagesHashed(img1, img2);
  std::cout << "Similarity: " << similarity << std::endl;
  std::cout << "Similarity (hashed): " << similarityHashed << std::endl;
  return 0;
}
