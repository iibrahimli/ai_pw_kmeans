#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <tuple>

#include "cls_sample.hpp"


namespace dt {


/*
    parses a file (for now) into a vector of cls_samples
*/
class parser {

public:

    // default constructor
    parser() = default;


    // parse csv file
    // provide dataset format using template arguments
    // ignore_first_col: pass true if the csv contains column names in the first row
    // returns the vector of samples and map from class id to class name (string) for the dataset
    template <typename feat_t, std::size_t n_feat>
    std::tuple<std::vector<cls_sample<feat_t, n_feat>>, label_map>
    parse_csv(const std::string & filename, char delim = ',', bool ignore_first_col = false){
        
        std::ifstream                            infile(filename);
        std::string                              line;
        std::string                              cell;
        std::istringstream                       line_ss;
        std::istringstream                       cell_ss;
        std::string                              label;
        unsigned int                             label_id;
        std::size_t                              line_n = 0;
        std::map<std::string, unsigned int>      label_to_id;
        unsigned int                             max_label_id = 0;

        std::array<feat_t, n_feat>               feats;
        std::vector<cls_sample<feat_t, n_feat>>  dataset;
        label_map                                dict;
        
        if(!infile)
            throw std::runtime_error("Couldn't open file");

        while(std::getline(infile, line)){

            if(line_n == 0 && ignore_first_col){
                ++line_n;
                line_ss.clear();
                continue;
            }

            line_ss.str(line);

            // parse features
            for(std::size_t cell_n = 0; cell_n < n_feat; ++cell_n){

                if(!line_ss)
                    throw std::runtime_error("Unexpected end of line");
                
                std::getline(line_ss, cell, delim);

                // parse cell
                cell_ss.str(cell);
                cell_ss >> feats[cell_n];
                cell_ss.clear();
            }

            // parse label
            if(!line_ss)
                throw std::runtime_error("Unexpected end of line");
            
            line_ss >> label;

            // add new label if its not in the seen labels map
            if(label_to_id.find(label) == label_to_id.end()){
                label_id = max_label_id;
                dict[max_label_id] = label;
                label_to_id.insert({label, max_label_id});
                ++max_label_id;
            }
            else{
                label_id = label_to_id[label];
            }
            
            // add sample to dataset, etc
            dataset.emplace_back(feats, label_id);
            ++line_n;
            line_ss.clear();
        }

        return {dataset, dict};
    }

};


} // namespace dt