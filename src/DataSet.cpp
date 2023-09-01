
#include <fstream>

#include "Dataset.h"

using namespace Eigen;

int getHeaderInt(std::ifstream& file) {
    int num = 0;
    char* buffer = new char[4];
    file.read(buffer, 4);
    std::memcpy((char*)&num, buffer + 3, 1);
    std::memcpy((char*)&num + 1, buffer + 2, 1);
    std::memcpy((char*)&num + 2, buffer + 1, 1);
    std::memcpy((char*)&num + 3, buffer, 1);
    delete buffer;
    return num;
}

DataSet::DataSet(std::string imagesFileName, std::string labelsFileName) {

    printf("Loading dataset from %s ... ", imagesFileName.c_str());

    try {
        std::ifstream imagesFile(imagesFileName, std::ios::in | std::ios::binary);
        std::ifstream labelsFile(labelsFileName, std::ios::in | std::ios::binary);

        if (!imagesFile.is_open() || !labelsFile.is_open()) {
            printf("Error: could not open dataset file\n");
            throw;
        }

        getHeaderInt(imagesFile);
        _imagesNum = getHeaderInt(imagesFile);
        _imgWidth = getHeaderInt(imagesFile);
        _imgHeight = getHeaderInt(imagesFile);

        getHeaderInt(labelsFile);
        int labelsNum = getHeaderInt(labelsFile);

        if (_imagesNum != labelsNum) {
            printf("Error: size mismatch between image and lables files\n");
            printf("%i images, %i labels", _imagesNum, labelsNum);
            throw;
        }

        char* imgBuffer = new char[_imagesNum * _imgWidth * _imgHeight];
        char* labelBuffer = new char[_imagesNum];

        imagesFile.read(imgBuffer, _imagesNum * _imgWidth * _imgHeight);
        labelsFile.read(labelBuffer, _imagesNum);

        imagesFile.close();
        labelsFile.close();

        _dataSet.resize(_imagesNum);

        for (int i = 0; i < _imagesNum; i++) {
            DataCase* dataCase = new DataCase(_imgWidth * _imgHeight, 10);
            dataCase->img = Map<Matrix<uint8_t, Dynamic, Dynamic, RowMajor>>((uint8_t*)(imgBuffer + i * _imgWidth * _imgHeight), _imgWidth, _imgHeight).reshaped().cast<double>() / 255.0;
            dataCase->label(labelBuffer[i]) = 1.0;
            _dataSet[i] = dataCase;
        }
        delete labelBuffer;
        delete imgBuffer;

        printf("done\n");

    } catch (...) {
        printf("Failed reading dataset from %s and %s", imagesFileName.c_str(), labelsFileName.c_str());
    }
}

