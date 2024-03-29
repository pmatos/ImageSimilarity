#include <bitset>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
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
    if (i < size && j < size && i != j && j > i)
      return data[j * (j - 1) / 2 + i];
    else {
      // swap i and j and call again the operator
      return (*this)(j, i);
    }
  }

  // Access element (i, j) in the lower triangular matrix (const version)
  const T &operator()(size_t i, size_t j) const {
    if (i < size && j < size && i != j && j > i)
      return data[j * (j + 1) / 2 + i];
    else {
      // swap i and j and call again the operator
      return (*this)(j, i);
    }
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
    Resolution = ImgData.total();
    ImgData.release();
  }

  ~Image() { Path.clear(); }

  inline uint64_t getHash() const { return Hash; }
  inline const std::string &getPath() const { return Path; }
  inline const std::string getExtension() const {
    std::filesystem::path p(Path);
    return p.extension().string();
  }
  inline const std::string getFilename() const {
    std::filesystem::path p(Path);
    return p.filename().string();
  }

  inline bool empty() const { return isEmpty; }
  inline size_t getResolution() const { return Resolution; }

private:
  bool isEmpty;
  std::string Path;
  uint64_t Hash;
  size_t Resolution;
};

int hammingDistance(uint64_t hash1, uint64_t hash2) {
  return std::bitset<64>(hash1 ^ hash2).count();
}

double compareImagesHashed(const Image &img1, const Image &img2) {
  if (img1.empty() || img2.empty()) {
    return 0.0;
  }

  int distance = hammingDistance(img1.getHash(), img2.getHash());

  // Normalize the Hamming distance to a similarity score in the range [0,
  // 1].
  return 1.0 - static_cast<double>(distance) / 64.0;
}

bool isImageFile(const std::string &filename) {
  std::vector<std::string> extensions = {".jpg",  ".jpeg", ".png", ".bmp",
                                         ".tiff", ".tif",  ".gif"};
  std::filesystem::path p(filename);
  std::filesystem::path ext = p.extension();

  // Case-insensitive comparison of the extension.
  std::string extStr = ext.string();
  transform(extStr.begin(), extStr.end(), extStr.begin(), ::tolower);
  for (const auto &ext : extensions) {
    if (ext == extStr) {
      return true;
    }
  }

  std::cout << "Warning: Not an image file: " << filename << "\n";
  return false;
}

void sortImageVec(std::vector<int> &imageInd,
                  const std::vector<Image> &imageFiles) {
  // sort images by how relevant they are. These are all duplicates.
  if (imageInd.size() <= 1)
    return; // nothing to do

  // If there is more than 1 image, then sort them by interestingness:
  // 1. Higher resolution images are more interesting.
  // 2. If two images have the same resolution, the one with more metadata
  // is more interesting.
  // 3. If two images have the same resolution and metadata, the one with
  // the lower index is more interesting. (arbitrary)
  std::sort(imageInd.begin(), imageInd.end(), [&imageFiles](int i1, int i2) {
    const Image &img1 = imageFiles[i1];
    const Image &img2 = imageFiles[i2];
    if (img1.getResolution() != img2.getResolution()) {
      return img1.getResolution() > img2.getResolution();
    } else {
      auto metadata1 = extractMetadata(img1.getPath());
      auto metadata2 = extractMetadata(img2.getPath());
      if (metadata1.size() != metadata2.size()) {
        return metadata1.size() > metadata2.size();
      } else {
        return i1 < i2;
      }
    }
  });
}

void printSimilaritySets(const std::string &path, const std::string &out,
                         double threshold) {
  // List all image files under the given path.
  std::vector<Image> initImageFiles;
  {
    std::filesystem::create_directories(out + "/non-image");
    size_t nonimgCount = 0;
    TIME_BLOCK("File Loading");
    for (const auto &entry : fs::recursive_directory_iterator(
             path, fs::directory_options::skip_permission_denied)) {
      if (entry.is_regular_file() && !fs::is_symlink(entry) &&
          isImageFile(entry.path().string())) {
        if (initImageFiles.size() % 100 == 0)
          std::cout << "Loaded " << initImageFiles.size() << " image files.\n";
        initImageFiles.emplace_back(entry.path().string());
      } else if (entry.is_regular_file() && !fs::is_symlink(entry)) {
        // copy file to output folder out/non-image
        std::filesystem::create_hard_link(
            entry.path().string(),
            out + "/non-image/" + "nonimage" + std::to_string(nonimgCount) +
                entry.path().filename().extension().string());
        nonimgCount++;
      }
    }
    std::cout << "Found " << initImageFiles.size() << " image files.\n";
  }

  // For debugging lets sort the files by path in the vector.
  std::sort(initImageFiles.begin(), initImageFiles.end(),
            [](const Image &img1, const Image &img2) {
              return img1.getPath() < img2.getPath();
            });

  const std::vector<Image> &imageFiles = initImageFiles;

  // Ensure no duplicates in imageFiles.
  std::set<std::string> imageSet;
  for (auto &Img : imageFiles) {
    imageSet.insert(Img.getPath());
  }
  assert(imageSet.size() == imageFiles.size());

  // List all file paths found to stdout
  {
    TIME_BLOCK("File Listing");
    for (size_t i = 0; i < imageFiles.size(); ++i) {
      std::cout << "* [" << i << "]: " << imageFiles[i].getPath() << "\n";
    }
  }

  // Compare every image with every other and fill in the TMatrix
  // ISSUE:
  // There is a problem with the way we are doing things. Two images can be
  // marked as duplicates of a third image, but not as duplicates of each other.
  // dup(A, X) and dup(B, X) does not imply dup(A, B).
  TMatrix<double> M(imageFiles.size());
  for (size_t i = 0; i < imageFiles.size() - 1; ++i) {
    const auto &img1 = imageFiles[i];
    TIME_BLOCK("File Analysis: " + img1.getPath());

    for (size_t j = i + 1; j < imageFiles.size(); ++j) {
      const auto &img2 = imageFiles[j];
      double similarity = compareImagesHashed(img1, img2);
      assert(similarity == compareImagesHashed(img2, img1));
      M(i, j) = similarity;
    }
  }

  // output matrix as csv
  {
    TIME_BLOCK("Output Matrix");
    std::ofstream csv(out + "/similarity.csv");
    // output header
    csv << ",";
    for (size_t i = 0; i < imageFiles.size(); ++i) {
      csv << imageFiles[i].getFilename() << ",";
    }
    csv << "\n";

    for (size_t i = 0; i < imageFiles.size(); ++i) {
      csv << imageFiles[i].getFilename() << ",";
      for (size_t j = 0; j < imageFiles.size(); ++j) {
        if (j <= i)
          csv << ",";
        else
          csv << M(i, j) << ",";
      }
      csv << "\n";
    }
  }

  // an image index is marked as done if it was already processed
  std::vector<bool> imageDone(imageFiles.size(), false);

  // Print similarities and copy to output folder
  {
    TIME_BLOCK("Image Analysis");
    for (size_t i = 0; i < imageFiles.size(); i++) {
      // Print loop headers and done images
      std::cout << "Loop start. Images done: ";
      for (size_t j = 0; j < imageDone.size(); j++) {
        if (imageDone[j]) {
          std::cout << j << " ";
        }
      }
      std::cout << std::endl;

      std::cout << "Checking if image " << i << " is done.\n";
      if (imageDone[i]) {
        std::cout << "= Seen image " << imageFiles[i].getPath() << " [" << i
                  << "]"
                  << " before.\n"; // this image has already been processed
        continue;
      }
      std::cout << "Not done - initiating processing\n";

      // vector containing indices of images similar to image i
      std::vector<int> imageInd;
      imageInd.push_back(i);

      for (size_t j = 0; j < imageFiles.size(); j++) {
        if (i == j)
          continue; // skip diagonal
        if (M(i, j) > threshold) {
          imageInd.push_back(j);
        }
      }

      // sort images by similarity
      sortImageVec(imageInd, imageFiles);

      // print imageInd
      std::cout << "Similar images to " << i << ": ";
      for (size_t j = 0; j < imageInd.size(); j++) {
        std::cout << imageInd[j] << " ";
      }
      std::cout << std::endl;

      // Remove any images from imageInd that have already been processed.
      for (auto it = imageInd.begin(); it != imageInd.end();) {
        if (imageDone[*it]) {
          std::cout << "=== Removing image " << *it << " from imageInd.\n";
          it = imageInd.erase(it);
        } else {
          ++it;
        }
      }

      const size_t mainIdx = imageInd[0];
      const std::string &base = imageFiles[mainIdx].getFilename();
      const std::string &baseTarget = "image-" + std::to_string(mainIdx) +
                                      imageFiles[mainIdx].getExtension();
      // the first image is the most interesting one
      // copy it to the output and the others to the dups folder
      std::cout << "+ " << base << " (" + baseTarget + ")" << std::endl;
      std::filesystem::create_hard_link(imageFiles[mainIdx].getPath(),
                                        out + "/" + baseTarget);
      std::cout << "= Marking image " << imageFiles[mainIdx].getPath() << " ["
                << mainIdx << "]"
                << " as done.\n";
      imageDone[mainIdx] = true;

      if (imageInd.size() > 1) {
        std::string dupsFolder = out + "/dupsOf-" + std::to_string(mainIdx);
        // create dups folder and copy the rest of the dups there
        std::filesystem::create_directory(out + "/dupsOf-" +
                                          std::to_string(mainIdx));

        // Copy original to dups folder as well.
        std::filesystem::create_hard_link(imageFiles[mainIdx].getPath(),
                                          dupsFolder + "/ORIGINAL-" +
                                              baseTarget);

        for (size_t j = 1; j < imageInd.size(); j++) {
          const size_t dupIdx = imageInd[j];
          std::string dupPath = imageFiles[dupIdx].getPath();
          std::string dupTarget = "image-" + std::to_string(dupIdx) +
                                  imageFiles[dupIdx].getExtension();
          std::cout << "-> " << dupPath << " (" << dupTarget << "), "
                    << M(mainIdx, dupIdx) << std::endl;
          std::filesystem::create_hard_link(dupPath,
                                            dupsFolder + "/" + dupTarget);
          std::cout << "= Marking image " << imageFiles[dupIdx].getPath()
                    << " [" << dupIdx << "]"
                    << " as done.\n";
          imageDone[dupIdx] = true;
        }
      }
    }
  }

  {
    TIME_BLOCK("Verification");
    size_t count = 0;
    for (const auto &entry : fs::recursive_directory_iterator(
             out, fs::directory_options::skip_permission_denied)) {
      if (entry.is_regular_file() && !fs::is_symlink(entry) &&
          isImageFile(entry.path().string())) {
        count++;
      }
    }
    if (count != imageFiles.size()) {
      std::cerr << "ERROR: number of copied files do not match number of "
                   "source files analysed.\n";
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
    if (argc != 4) {
      std::cerr << "Error: Incorrect number of arguments for default mode.\n"
                << "Usage: ImageSimilarity threshold images_path out_path\n";
      return 1;
    }

    double threshold = std::stod(argv[1]);
    if (threshold < 0.0 || threshold > 1.0) {
      std::cerr << "Error: Threshold must be between 0 and 1.\n";
      return 1;
    }

    std::string path = argv[2];
    std::string out = argv[3];
    printSimilaritySets(path, out, threshold);
  }

  return 0;
}
