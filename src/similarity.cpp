#include <bitset>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

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

bool isImageFile(const std::string &filename) {
  std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp",
                                         ".tiff"};

  for (const auto &ext : extensions) {
    if (filename.size() >= ext.size() &&
        filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0) {
      return true;
    }
  }

  return false;
}

std::vector<std::set<std::string>> getSimilaritySets(const std::string &path,
                                                     double threshold) {
  std::vector<std::set<std::string>> similaritySets;

  // List all image files under the given path.
  std::vector<std::string> imageFiles;
  for (const auto &entry : fs::recursive_directory_iterator(
           path, fs::directory_options::skip_permission_denied)) {
    if (entry.is_regular_file() && !fs::is_symlink(entry) &&
        isImageFile(entry.path().string())) {
      imageFiles.push_back(entry.path().string());
    }
  }

  // Compare every pair of images.
  for (const auto &imgPath1 : imageFiles) {
    cv::Mat img1 = cv::imread(imgPath1);
    bool addedToSet = false;

    for (auto &similaritySet : similaritySets) {
      for (const auto &imgPath2 : similaritySet) {
        cv::Mat img2 = cv::imread(imgPath2);
        double similarity = compareImagesHashed(img1, img2);

        if (similarity >= threshold) {
          similaritySet.insert(imgPath1);
          addedToSet = true;
          break;
        }
      }

      if (addedToSet) {
        break;
      }
    }

    // If the image was not added to any existing set, create a new set for it.
    if (!addedToSet) {
      std::set<std::string> newSet = {imgPath1};
      similaritySets.push_back(newSet);
    }
  }

  return similaritySets;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: \n"
              << "Similarity mode: ImageSimilarity -s img1_path img2_path\n"
              << "Default mode: ImageSimilarity images_path\n";
    return 1;
  }

  std::string flag = argv[1];

  if (flag == "-s") {
    // Similarity mode
    if (argc != 4) {
      std::cerr << "Error: Incorrect number of arguments for similarity mode.\n"
                << "Usage: ImageSimilarity -s img1_path img2_path\n";
      return 1;
    }

    cv::Mat img1 = cv::imread(argv[2]);
    cv::Mat img2 = cv::imread(argv[3]);
    double similarityHashed = compareImagesHashed(img1, img2);
    std::cout << "Similarity (hashed): " << similarityHashed << std::endl;
  } else {
    // Default mode
    if (argc != 3) {
      std::cerr << "Error: Incorrect number of arguments for default mode.\n"
                << "Usage: ImageSimilarity threshold images_path\n";
      return 1;
    }

    double threshold = std::stod(argv[1]);
    if (threshold < 0.0 || threshold > 1.0) {
      std::cerr << "Error: Threshold must be between 0 and 1.\n";
      return 1;
    }

    std::string path = argv[2];
    std::vector<std::set<std::string>> similaritySets =
        getSimilaritySets(path, threshold);

    // Print the similarity sets
    int setIndex = 1;
    for (const auto &similaritySet : similaritySets) {
      std::cout << "Set " << setIndex++ << ":" << std::endl;
      for (const auto &imgPath : similaritySet) {
        std::cout << "  " << imgPath << std::endl;
      }
    }
  }

  return 0;
}
