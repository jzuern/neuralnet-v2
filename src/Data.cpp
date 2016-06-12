#include "Data.h"

Data::Data(){
        // empty constructor
}

void Data::load_data_from_file(const std::string filename){

        std::ifstream file(filename);
        std::string line;

        nEntries = 0;
        while(std::getline(file,line) ) {

                int idx = 0;
                VectorXd image(28*28);
                // format: label, pix-11, pix-12, pix-13, ...
                //   where pix-ij is the pixel in the ith row and jth column.
                std::stringstream lineStream(line);
                std::string cell;
                while(std::getline(lineStream,cell,','))
                {
                        // accessing single cell
                        if (idx == 0) { //first entry is classifier
                                correctDigit.push_back(stoi(cell));
                        }
                        else{ // remaing entries are img data
                                double greyValue = stod(cell)/255.0;
                                image[idx-1] = greyValue;
                        }
                        idx += 1;
                }
                nEntries += 1;
                imgData.push_back(image);

                if(nEntries % 10000 == 0) std::cout << "    Allocated " << nEntries << " data sets" << std::endl;
        }
        std::cout << "Allocated all data sets" << std::endl;
}

int Data::getDigitEntry(int index){
        return correctDigit[index];
}

VectorXd Data::getImgEntry(int index){
        return imgData[index];
}
