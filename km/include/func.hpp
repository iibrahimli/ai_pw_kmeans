/*
    functions used in decision trees
*/

#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "cls_sample.hpp"
#include "parser.hpp"



/*
    printing a vector of primitives lmao
*/
template <typename T>
std::ostream & operator << (std::ostream & out, std::vector<T> vec){
    
    out << "[ ";
    for(auto i = 0ul; i < vec.size(); ++i){
        out << vec[i];
        if(i != vec.size() - 1)
            out << ", ";
        else
            out << " ";
    }
    out << "]";
    
    return out;
}


/*
    specialization for printing a dataset
*/
template <typename feat_t, std::size_t n_feat>
std::ostream & operator << (std::ostream & out, std::vector<dt::cls_sample<feat_t, n_feat>> vec){
    
    for(auto i = 0ul; i < vec.size(); ++i){
        out << vec[i];
        out << std::endl;
    }
    
    return out;
}


/*
    number of labels for a dataset (max_label_id + 1)
*/
template <typename feat_t, std::size_t n_feat>
unsigned int num_labels(const std::vector<dt::cls_sample<feat_t, n_feat>> & dataset){
    
    unsigned int num_lbl = 0;

    for(auto & s : dataset){
        if(s.label_id() > num_lbl)
            num_lbl = s.label_id();
    }

    return num_lbl + 1;
}


/*
    number of groups for a feature
    ASSUMING THE GROUP IDS ARE CONTIGUOUS & START FROM 0
*/
template <typename feat_t, std::size_t n_feat>
unsigned int num_groups(const std::vector<dt::cls_sample<feat_t, n_feat>> & dataset, std::size_t i){
    
    unsigned int num_grp = 0;

    for(auto & s : dataset){
        if(s[i] > num_grp)
            num_grp = s[i];
    }

    return num_grp + 1;
}


/*
    returns a vector of number of samples belonging to each label
*/
template <typename feat_t, std::size_t n_feat>
std::vector<unsigned int> num_per_label(const std::vector<dt::cls_sample<feat_t, n_feat>> & dataset){
    
    unsigned int num_lbl = num_labels(dataset);
    std::vector<unsigned int> lbl_count(num_lbl, 0);

    for(auto i = 0; i < dataset.size(); ++i){
        ++lbl_count[dataset[i].label_id()];
    }

    return lbl_count;
}


/*
    returns a vector of number of samples belonging to each group
*/
template <typename feat_t, std::size_t n_feat>
std::vector<unsigned int> num_per_group(const std::vector<dt::cls_sample<feat_t, n_feat>> & dataset, std::size_t i_feat){
    
    unsigned int num_grp = num_groups(dataset, i_feat);
    std::vector<unsigned int> grp_count(num_grp, 0);

    for(auto i = 0; i < dataset.size(); ++i){
        ++grp_count[dataset[i][i_feat]];
    }

    return grp_count;
}


/*
    majority of labels
*/
template <typename feat_t, std::size_t n_feat>
std::tuple<std::size_t, float> majority_label(const std::vector<dt::cls_sample<feat_t, n_feat>> & dataset){
    auto          npl      = num_per_label(dataset);
    std::size_t   ml_index = 0;

    for(auto i = 0lu; i < npl.size(); ++i){
        if(npl[i] > npl[ml_index])
            ml_index = i;
    }

    float ratio = (float) npl[ml_index] / dataset.size();

    return {ml_index, ratio};
}


/*
    get group as a separate dataset
*/
template <typename feat_t, std::size_t n_feat>
std::vector<dt::cls_sample<feat_t, n_feat>>
get_group_as_ds(const std::vector<dt::cls_sample<feat_t, n_feat>> & dataset, std::size_t i_feat, std::size_t group){
    
    std::vector<dt::cls_sample<feat_t, n_feat>> grp_ds;

    std::copy_if(dataset.cbegin(), dataset.cend(), std::back_inserter(grp_ds),
                [i_feat, group](const dt::cls_sample<feat_t, n_feat> & smpl){
                    return smpl[i_feat] == group;
                } );

    return grp_ds;
}


/*
    Shannon entropy of a dataset
*/
template <typename feat_t, std::size_t n_feat>
float dataset_entropy(std::vector<dt::cls_sample<feat_t, n_feat>> & dataset){
    
    float h = 0;
    float prob;

    auto total = dataset.size();
    auto counts = num_per_label(dataset);

    for(auto i = 0; i < counts.size(); ++i){
        if(counts[i] == 0) continue;
        prob = (float) counts[i] / total;
        h -= prob * std::log2(prob);
    }

    return h;
}


/*
    Shannon entropy of a group
*/
template <typename feat_t, std::size_t n_feat>
float group_entropy(const std::vector<dt::cls_sample<feat_t, n_feat>> & dataset, std::size_t i_feat, std::size_t group){
    
    auto grp_ds = get_group_as_ds(dataset, i_feat, group);

    float h = dataset_entropy(grp_ds);
    
    return h;
}


/*
    Discriminative power of attribute i_feat
*/
template <typename feat_t, std::size_t n_feat>
float disc(const std::vector<dt::cls_sample<feat_t, n_feat>> & dataset, std::size_t i_feat){
    
    // UNNECESSARY ???
    // auto ds = sort_dataset(dataset, i_feat);
    auto ds = dataset;

    // calculate dataset entropy
    float dp = dataset_entropy(ds);

    std::size_t n_groups = num_groups(ds, i_feat);
    auto num_per_grp = num_per_group(ds, i_feat);

    // calculate discriminative power
    for(auto grp = 0ul; grp < n_groups; ++grp)
        dp -= (float) num_per_grp[grp] / ds.size() * group_entropy(ds, i_feat, grp);
    
    return dp;
}


/*
    get id of the feature with the highest discriminative power
*/
template <typename feat_t, std::size_t n_feat>
std::size_t get_best_feature(const std::vector<dt::cls_sample<feat_t, n_feat>>& dataset){
    
    std::size_t best_feat = 0;
    float       best_disc = 0;

    for(std::size_t bf = 0; bf < n_feat; ++bf){
        if(disc(dataset, bf) > best_disc){
            best_feat = bf;
        }
    }
    
    return best_feat;
}


/*
    sort the dataset by attribute i, where 0 <= i <= n_features
    (!) returns a sorted copy
*/
template <typename feat_t, std::size_t n_feat>
std::vector<dt::cls_sample<feat_t, n_feat>> sort_dataset(const std::vector<dt::cls_sample<feat_t, n_feat>>& dataset, std::size_t i){

    if(i >= n_feat)
        throw std::runtime_error("invalid attribute id");
    
    auto ds = dataset;

    std::stable_sort(ds.begin(), ds.end(),
                    [i](const dt::cls_sample<feat_t, n_feat> & a, const dt::cls_sample<feat_t, n_feat> & b) {
                        return a[i] < b[i];
                    });

    return ds;
}


/*
    discretize the dataset, dividing each feature into n_groups groups
    (!) returns a discretized copy
*/
template <typename feat_t, std::size_t n_feat>
std::vector<dt::cls_sample<feat_t, n_feat>> discretize_dataset(const std::vector<dt::cls_sample<feat_t, n_feat>>& dataset, std::size_t n_groups){

    auto ds  = dataset;
    auto spg = ds.size() / n_groups;    // samples per group

    for(auto feat = 0ul; feat < n_feat; ++feat){

        // sort by feature
        ds = sort_dataset(ds, feat);

        for(int s = 0; s < ds.size(); ++s){
            ds[s][feat] = std::floor(s / spg);
        }
    }

    return ds;
}


/*
    make train/test split
*/
template <typename feat_t, std::size_t n_feat>
std::tuple<std::vector<dt::cls_sample<feat_t, n_feat>>,
           std::vector<dt::cls_sample<feat_t, n_feat>>>
split_dataset(const std::vector<dt::cls_sample<feat_t, n_feat>>& dataset, float ratio, bool shuffle = true){

    auto ds = dataset;

    if(ratio < 0 || ratio > 1.0)
        throw std::invalid_argument("ratio must be between 0.0 and 1.0");

    if(shuffle)
        std::random_shuffle(ds.begin(), ds.end());
    
    std::size_t index = std::round(dataset.size() * ratio);

    decltype(ds) train {ds.cbegin(),             ds.cbegin() + index + 1};
    decltype(ds) test  {ds.cbegin() + index + 1, ds.cend()};

    return {train, test};
}