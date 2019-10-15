#pragma once

#include <vector>

#include "cls_sample.hpp"
#include "func.hpp"


namespace dt{


enum node_type {
    BRANCH,
    LEAF
};


struct decision_tree_node {

    node_type                           type;         // type of node
    std::size_t                         feat_id;      // [BRANCH] index of feature to consider
    std::size_t                         group_id;     //          group in i_feat to consider
    std::size_t                         pred;         // [LEAF]   prediction (class_id) if it is a LEAF node
    std::vector<decision_tree_node *>   branches;     // [BRANCH] branches corresponding to each group
    std::size_t                         depth;        // depth of node in tree


    decision_tree_node()
        :   type     {LEAF},
            feat_id  {0},
            group_id {0},
            pred     {0},
            depth    {0}
    {}


    decision_tree_node(node_type tp, std::size_t feat_i = 0, std::size_t group_i = 0, std::size_t prediction = 0, std::size_t depth = 0)
        :   type     {tp},
            feat_id  {feat_i},
            group_id {group_i},
            pred     {prediction},
            depth    {depth}
    {}

};


struct decision_tree{

private:

    // root node
    decision_tree_node * root;

    // number of groups for all features (assumend to be the same)
    std::size_t          n_groups;

    // minimum samples (stopping condition)
     std::size_t         min_samples;

    // max depth (stopping condition)
    std::size_t          max_depth;



    /*
        split and add nodes
    */
    template <typename feat_t, std::size_t n_feat>
    void split(decision_tree_node * cur_node, const std::vector<cls_sample<feat_t, n_feat>> & dataset){

        auto best_feature = get_best_feature(dataset);

        // check stopping condition
        if(cur_node->depth > max_depth || dataset.size() <= min_samples){
            // time to stop

            auto [ml, _] = majority_label(dataset);

            // make this node a leaf
            cur_node->type = LEAF;
            cur_node->pred = ml;
        }
        else{

            // this node is a branch
            cur_node->type = BRANCH;

            // add child nodes for each group
            for(int g = 0; g < n_groups; ++g){
                cur_node->branches.emplace_back(new decision_tree_node(
                    LEAF,
                    best_feature,
                    g,
                    0,
                    cur_node->depth + 1
                ));
            }

            // split child nodes
            for(auto nd : cur_node->branches)
                split(nd, get_group_as_ds(dataset, nd->feat_id, nd->group_id));
        }
    }


public:

    /*
        default constructor
        threshold: stopping condition threshold
                   default sc is (#majority_label / #group_size) > threshold
    */
    decision_tree(std::size_t max_depth = 5, std::size_t min_samples = 3)
        :   root         {nullptr},
            n_groups     {0},
            min_samples  {min_samples},
            max_depth    {max_depth}
    {}


    /*
        builds a decision tree given a dataset (vector of cls_samples)
    */
    template <typename feat_t, std::size_t n_feat>
    void fit(const std::vector<cls_sample<feat_t, n_feat>> & dataset){

        auto ds = dataset;
        std::vector<float> feat_disc(n_feat, 0);
        
        // ASSUMING NUMBER OF GROUPS IS SAME FOR ALL FEATURES
        n_groups = num_groups(ds, 0);
        
        // compute discriminative power for each feature
        for(auto ftr = 0lu; ftr < n_feat; ++ftr)
            feat_disc[ftr] = disc(ds, ftr);
        
        // make list of features from most disc to least
        std::vector<std::size_t> features;
        auto                     feat_tmp = feat_disc;

        // sort features
        for(int i = 0; i < feat_tmp.size(); ++i){
            std::size_t max_index = std::distance(feat_tmp.begin(), std::max_element(feat_tmp.begin(), feat_tmp.end()));
            features.emplace_back(max_index);
            feat_tmp[max_index] = 0;
        }

        // --- SPLITTING ---

        root = new decision_tree_node(BRANCH);

        // recursively split nodes until stopping condition is met
        split(root, ds);

    }


    /*
        returns predicted class_id
    */
    template <typename feat_t, std::size_t n_feat>
    std::size_t predict(const cls_sample<feat_t, n_feat> & sample){

        auto         cur_node = root;
        std::size_t  feat;
        std::size_t  grp;

        while(cur_node->type != LEAF){
            feat = cur_node->feat_id;
            cur_node = cur_node->branches[sample[feat]];
        }

        return cur_node->pred;
    }


    /*
        builds a decision tree given a dataset (vector of cls_samples)
    */
    template <typename feat_t, std::size_t n_feat>
    void evaluate(const std::vector<cls_sample<feat_t, n_feat>> & dataset){

        std::vector<std::size_t> gt;
        std::vector<std::size_t> pred;
        int correct_pred = 0;

        for(int i = 0; i < dataset.size(); ++i){
            gt.emplace_back(dataset[i].label_id());
            pred.emplace_back(predict(dataset[i]));

            if(gt[i] == pred[i]) ++correct_pred;
        }

        std::cout << std::fixed << std::left << std::setw(25) << "test samples: " << std::setw(5) << dataset.size() << std::endl;
        std::cout << std::fixed << std::left << std::setw(25) << "correct predictions: " << std::setw(5) << correct_pred << std::endl;
        std::cout << std::fixed << std::left << std::setw(25) << "accuracy: " << std::setw(5) << std::setprecision(2) << (float) correct_pred / dataset.size() << std::endl;

    }


};


}  // namespace dt