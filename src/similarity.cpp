#include <bitset>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class Timer {
public:
  Timer(const std::string &name)
      : m_Name(name), m_Start(std::chrono::high_resolution_clock::now()) {}

  ~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - m_Start;
    std::cout << "Block " << m_Name << " took: " << elapsed.count() << "ms\n";
  }

private:
  std::string m_Name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
};

#define TIME_BLOCK(name) Timer timer##__LINE__(name)

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

class Image {
public:
  Image(const std::string &Path) : Path(Path), isEmpty(false) {
    cv::Mat ImgData = cv::imread(Path);
    if (ImgData.empty()) {
      std::cerr << "Warning: Could not load image " << Path << "\n";
      Hash = 0;
      isEmpty = true;
    } else
      Hash = computeDHash(ImgData);
    ImgData.release();
  }

  ~Image() { Path.clear(); }

  inline uint64_t getHash() const { return Hash; }
  inline const std::string &getPath() const { return Path; }
  inline bool empty() const { return isEmpty; }

private:
  bool isEmpty;
  std::string Path;
  uint64_t Hash;
};

int hammingDistance(uint64_t hash1, uint64_t hash2) {
  return std::bitset<64>(hash1 ^ hash2).count();
}

double compareImagesHashed(const Image &img1, const Image &img2) {
  if (img1.empty() || img2.empty()) {
    return 0.0;
  }

  int distance = hammingDistance(img1.getHash(), img2.getHash());

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

std::vector<std::vector<std::string>> getSimilaritySets(const std::string &path,
                                                        double threshold) {
  std::vector<std::vector<std::string>> similaritySets;

  // List all image files under the given path.
  std::vector<Image> imageFiles;
  for (const auto &entry : fs::recursive_directory_iterator(
           path, fs::directory_options::skip_permission_denied)) {
    if (entry.is_regular_file() && !fs::is_symlink(entry) &&
        isImageFile(entry.path().string())) {
      if (imageFiles.size() % 100 == 0)
        std::cout << "Loaded " << imageFiles.size() << " image files.\n";
      imageFiles.emplace_back(entry.path().string());
    }
  }
  std::cout << "Found " << imageFiles.size() << " image files.\n";

  // Compare every pair of images.
  for (const auto &img1 : imageFiles) {
    bool addedToSet = false;

    // Instead of comparing imgPath1 with all images in the set, we
    // compare it with just a random image in the set.
    for (auto &similaritySet : similaritySets) {
      size_t sz = similaritySet.size();
      size_t randIndex = std::rand() % sz;
      auto it = similaritySet.begin();
      std::advance(it, randIndex);
      const auto &img2 = *it;
      double similarity = compareImagesHashed(img1, img2);

      if (similarity >= threshold) {
        similaritySet.emplace_back(img1.getPath());
        addedToSet = true;
        break;
      }
    }

    // If the image was not added to any existing set, create a new set for
    // it.
    if (!addedToSet) {
      std::vector<std::string> newSet;
      newSet.push_back(img1.getPath());
      similaritySets.emplace_back(newSet);
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

    Image img1(argv[2]);
    Image img2(argv[3]);
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
    auto similaritySets = getSimilaritySets(path, threshold);

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
