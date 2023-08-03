#include <bitset>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <libexif/exif-data.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

template <typename T> class TMatrix {
private:
  std::vector<T> data;
  size_t size;

public:
  TMatrix(size_t n) : size(n) {
    // Calculate the number of elements needed to store the lower triangular
    // matrix without diagonal.
    data.resize(n * (n - 1) / 2);
  }

  // Access element (i, j) in the lower triangular matrix
  T &operator()(size_t i, size_t j) {
    assert(i < size && j < size && i != j && j > i);

    return data[j * (j - 1) / 2 + i];
  }

  // Access element (i, j) in the lower triangular matrix (const version)
  const T &operator()(size_t i, size_t j) const {
    assert(i < size && j < size && i != j && j > i);

    return data[j * (j + 1) / 2 + i];
  }
};

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

std::map<std::string, std::string> extractMetadata(const std::string &path) {
  std::map<std::string, std::string> metadata;

  ExifData *exifData = exif_data_new_from_file(path.c_str());
  if (exifData) {
    for (int i = 0; i < EXIF_IFD_COUNT; ++i) {
      ExifContent *content = exifData->ifd[i];
      if (content) {
        exif_content_foreach_entry(
            content,
            [](ExifEntry *entry, void *user_data) {
              char buf[1024];
              if (exif_entry_get_value(entry, buf, sizeof(buf)) &&
                  buf[0] != '\0') {
                std::string tag_name = exif_tag_get_name(entry->tag);
                auto &metadata_map =
                    *reinterpret_cast<std::map<std::string, std::string> *>(
                        user_data);
                metadata_map[tag_name] = buf;
              }
            },
            &metadata); // Pass the address of metadata as user_data
      }
    }
    exif_data_unref(exifData);
  }

  return metadata;
}

void compareMetadata(const std::string &path1, const std::string &path2) {
  auto metadata1 = extractMetadata(path1);
  auto metadata2 = extractMetadata(path2);

  std::cout << "Comparing metadata between " << path1 << " and " << path2
            << ":\n";

  for (const auto &[key, value] : metadata1) {
    if (metadata2.find(key) != metadata2.end() && metadata2[key] != value) {
      std::cout << "  Mismatch in " << key << ": " << path1 << " has " << value
                << ", " << path2 << " has " << metadata2[key] << "\n";
    }
  }

  int count1 = metadata1.size();
  int count2 = metadata2.size();

  if (count1 > count2)
    std::cout << path1 << " has more complete metadata.\n";
  else if (count2 > count1)
    std::cout << path2 << " has more complete metadata.\n";
  else
    std::cout << "Both files have the same amount of metadata.\n";
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

void printSimilaritySets(const std::string &path, double threshold) {

  // List all image files under the given path.
  std::vector<Image> imageFiles;
  {
    TIME_BLOCK("File Loading");
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
  }

  // Compare every image with a random image in the existing sets.
  TMatrix<double> M(imageFiles.size());
  for (size_t i = 0; i < imageFiles.size() - 1; ++i) {
    const auto &img1 = imageFiles[i];
    TIME_BLOCK("File Analysis: " + img1.getPath());

    for (size_t j = i + 1; j < imageFiles.size(); ++j) {
      const auto &img2 = imageFiles[j];
      double similarity = compareImagesHashed(img1, img2);
      M(i, j) = similarity;
    }
  }

  // Print similarities
  {
    TIME_BLOCK("Display Similarities");
    for (size_t i = 0; i < imageFiles.size(); i++) {
      bool headerPrinted = false;
      for (size_t j = i + 1; j < imageFiles.size(); j++) {
        if (M(i, j) > threshold) {
          if (!headerPrinted) {
            std::cout << imageFiles[i].getPath() << std::endl;
            headerPrinted = true;
          }
          std::cout << "    =" << imageFiles[j].getPath() << " : " << M(i, j)
                    << std::endl;
          compareMetadata(imageFiles[i].getPath(), imageFiles[j].getPath());
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: \n"
              << "Similarity mode: ImageSimilarity -s img1_path img2_path\n"
              << "Default mode: ImageSimilarity <threshold> images_path\n";
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
    printSimilaritySets(path, threshold);
  }

  return 0;
}
