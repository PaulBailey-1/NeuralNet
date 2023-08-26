
#include <string>
#include <iostream>

#include <Eigen/Dense>

class DataSet {
public:

    struct DataCase {
        // Eigen::MatrixXd img;
        Eigen::VectorXd img;
        Eigen::VectorXd label;
    };

    DataSet(std::string imageFileName, std::string labelFileName);

    int getSize() {return _imagesNum;}
    int getImgWidth() {return _imgWidth;}
    int getImgHeight() {return _imgHeight;}

    const Eigen::VectorXd& getImg(int i) {
        return _dataSet[i]->img;
    }

    const Eigen::VectorXd& getLabel(int i) {
        return _dataSet[i]->label;
    }

private:

    int _imagesNum{0};
    int _imgWidth{0};
    int _imgHeight{0};

    std::vector<DataCase*> _dataSet;

};