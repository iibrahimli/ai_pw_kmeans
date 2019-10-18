#pragma once

#include <vector>
#include <set>
#include <random>
#include <algorithm>

#include "cls_sample.hpp"
#include "func.hpp"


namespace km{


struct k_means{

private:

    int                                k;            // number of clusters

    std::vector<std::vector<double>>   centroids;    // centroids

    int                                max_iter;     // maximum number of iterations

    double                             delta;        // max diff of centroid between iterations

    double                             min_delta;    // stopping condition

    std::string                        init_method;  // "random" or "kmeans++"



    /*
        return index of the closest centroid to p
    */
    template <typename feat_t, std::size_t n_feat>    
    std::size_t closest_centroid(const cls_sample<feat_t, n_feat> & p){

        std::size_t   closest;
        double        cl_dist = std::numeric_limits<double>::infinity();
        double        tmp_dist;

        for(std::size_t cl = 0; cl < centroids.size(); ++cl){
            if((tmp_dist = euclidean_dist(p, centroids[cl])) < cl_dist){
                closest = cl;
                cl_dist = tmp_dist;
            }
        }

        return closest;
    }


    /*
        helper function to calculate co-occurence of gt_label, pred_label pair
    */
    int get_cooccurence(const std::vector<std::size_t> & gt, const std::vector<std::size_t> & pred, std::size_t gt_l, std::size_t pred_l){
        int cooc = 0;

        for(int i = 0; i < gt.size(); ++i){
            if(gt[i] == gt_l &&  pred[i] == pred_l)
                ++cooc;
        }
        
        return cooc;
    }


    /*
        helper function to choose best match between clusters and ground truth labels
    */
    std::vector<std::size_t> get_match(const std::vector<std::size_t> & gt, const std::vector<std::size_t> & pred){
        std::vector<std::size_t> match(k, 0);
        int                      max_count     = 0;
        std::size_t              best_gt_label = 0;
        int                      cooc;

        // get co-occurences
        for(int pred_label = 0; pred_label < k; ++pred_label){
            max_count     = 0;
            best_gt_label = 0;
            for(int gt_label = 0; gt_label < k; ++gt_label){
                
                cooc = get_cooccurence(gt, pred, gt_label, pred_label);
                
                if(cooc > max_count){
                    max_count = cooc;
                    best_gt_label = gt_label;
                }
                
                // std::cout << cooc << " ";
            }

            // std::cout << std::endl;

            match[pred_label] = best_gt_label;
        }

        // std::cout << "match: " << match << std::endl;

        return match;
    }


public:


    /*
        k:               number of clusters
        max_iter:        maximum number of iterations (-1 = unlimited)
        init_method:     initialization method ("k++", "random")
    */
    k_means(int k, int max_iter = 10000, double min_delta = 0.001, std::string init_method = "kmeans++")
        :   k           {k},
            max_iter    {max_iter},
            delta       {min_delta + 1},
            min_delta   {min_delta},
            init_method {init_method}
    {
        if(init_method != "kmeans++" && init_method != "random")
            throw std::invalid_argument("init_method must be one of: \"kmeans++\", \"random\"");
    }


    /*
        fits K-Means on the given dataset (vector of cls_samples)
    */
    template <typename feat_t, std::size_t n_feat>
    int fit(const std::vector<cls_sample<feat_t, n_feat>> & dataset){

        if(dataset.size() <= k || num_labels(dataset) > k)
            throw std::runtime_error("why on earth would you cluster something like this...");

        // choose initial clusters

        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> uni(0, dataset.size());

        if(init_method == "kmeans++"){
            
            // first centroid - randomly chosen
            auto                 c_idx          = uni(rng);
            double               cdf            = 0;            // cumulative distribution
            std::vector<double>  weights(dataset.size(), 0);    // D^2 weights
            double               sum_dist       = 0;            // sum of distances to closest centroid

            // add first centroid
            centroids.emplace_back();
            centroids[centroids.size()-1].insert(
                centroids[centroids.size()-1].end(),
                std::begin(dataset[c_idx]._features),
                std::end(dataset[c_idx]._features)
            );

            // choose the rest of centroids
            for(int c = 0; c < k-1; ++c){

                // generate random number (used with cdf to choose centroids according to weights)
                double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

                // calculate distances to closest centroid for all data points and their sum
                for(c_idx = 0; c_idx < dataset.size(); ++c_idx){
                    weights[c_idx] = euclidean_dist(dataset[c_idx], centroids[closest_centroid(dataset[c_idx])]);
                    weights[c_idx] = weights[c_idx] * weights[c_idx];
                    sum_dist += weights[c_idx];
                }

                // 
                for(c_idx = 0; c_idx < dataset.size(); ++c_idx){

                    // calculate weight for sample c_idx
                    weights[c_idx] = weights[c_idx] / sum_dist;
                    cdf += weights[c_idx];

                    // choose according to D^2 probability
                    if(r < cdf){
                        if(c_idx != 0)
                            --c_idx;
                        break;
                    }
                }

                // add chosen sample to centroids
                centroids.emplace_back();
                centroids[centroids.size()-1].insert(
                    centroids[centroids.size()-1].end(),
                    std::begin(dataset[c_idx]._features),
                    std::end(dataset[c_idx]._features)
                );
            }

        }
        else{
            std::set<int> chosen_idx;

            for(int c = 0; c < k; ++c){
                auto c_idx = uni(rng);

                // check whether this index has already been picked
                if(chosen_idx.find(c_idx) == chosen_idx.end()){
                    chosen_idx.insert(c_idx);

                    // add chosen sample to centroids
                    centroids.emplace_back();
                    centroids[centroids.size()-1].insert(
                        centroids[centroids.size()-1].end(),
                        std::begin(dataset[c_idx]._features),
                        std::end(dataset[c_idx]._features)
                    );
                }

            }

        }


        int iter = 0;
        std::vector<std::size_t>   assigned_clusters(dataset.size(), 0);
        std::array<double, n_feat> mean;
        int                        n_points = 0;
        double                     cur_delta;
        double                     max_delta;

        for(; (max_iter == -1 || iter < max_iter) && delta > min_delta; ++iter){

            max_delta = 0;

            // if(iter % 5 == 0){
            //     std::cout << "iter " << iter << " - delta: " << delta << std::endl;
            //     for(int ci = 0; ci < centroids.size(); ++ci)
            //         std::cout << centroids[ci] << std::endl;
            //     std::cout << std::endl;
            // }

            // calculate which cluster each point belongs to
            for(int s_idx = 0; s_idx < dataset.size(); ++s_idx){
                assigned_clusters[s_idx] = closest_centroid(dataset[s_idx]);
            }

            // re-calculate centroids
            for(auto ki = 0; ki < k; ++ki){
                mean = {0};
                cur_delta = 0;
                n_points = 0;

                for(auto si = 0; si < dataset.size(); ++si){

                    // skip samples that do not belong to this cluster
                    if(assigned_clusters[si] != ki)
                        continue;

                    // calculate mean
                    for(auto fi = 0; fi < n_feat; ++fi)
                        mean[fi] = (mean[fi] * n_points + dataset[si][fi]) / (n_points + 1);
                    
                    ++n_points;
                }

                // calculate current delta
                for(auto fi = 0; fi < n_feat; ++fi)                
                    cur_delta += (centroids[ki][fi] - mean[fi]) * (centroids[ki][fi] - mean[fi]);
                cur_delta = sqrt(cur_delta);

                if(cur_delta > max_delta)
                    max_delta = cur_delta;

                // update centroid
                for(auto fi = 0; fi < n_feat; ++fi)
                    centroids[ki][fi] = mean[fi];
            }

            delta = max_delta;
        }

        return iter;
    }


    /*
        set cluster ids
        ex: k=3, [2, 0, 1]
    */
    void set_cluster_ids(std::vector<std::size_t> order){

        std::vector<std::vector<double>> new_centroids;  

        for(int i = 0; i < order.size(); ++i){
            new_centroids.emplace_back(centroids[order[i]]);
            // std::iter_swap(centroids.begin() + i, centroids.begin() + order[i]);
        }
        centroids = new_centroids;
    }
    
    
    /*
        returns predicted class_id
    */
    template <typename feat_t, std::size_t n_feat>
    std::size_t predict(const cls_sample<feat_t, n_feat> & sample){

        return closest_centroid(sample);
    }


    /*
        evaluates the clusters given a dataset (vector of cls_samples)
        NB: set_cluster_ids before calling this
    */
    template <typename feat_t, std::size_t n_feat>
    void evaluate(const std::vector<cls_sample<feat_t, n_feat>> & dataset){

        std::vector<std::size_t> gt;
        std::vector<std::size_t> pred;
        int correct_pred = 0;

        // predict on dataset
        for(int i = 0; i < dataset.size(); ++i){
            gt.emplace_back(dataset[i].label_id());
            pred.emplace_back(predict(dataset[i]));
        }

        // get best match
        auto bm = get_match(gt, pred);
        set_cluster_ids(bm);

        // evaluate
        gt.clear();
        pred.clear();
        for(int i = 0; i < dataset.size(); ++i){
            gt.emplace_back(dataset[i].label_id());
            pred.emplace_back(predict(dataset[i]));
            if(gt[i] == pred[i]) ++correct_pred;
        }

        std::cout << std::fixed << std::left << std::setw(25) << "samples: " << std::setw(5) << dataset.size() << std::endl;
        std::cout << std::fixed << std::left << std::setw(25) << "correct predictions: " << std::setw(5) << correct_pred << std::endl;
        std::cout << std::fixed << std::left << std::setw(25) << "accuracy: " << std::setw(5) << std::setprecision(2) << (float) correct_pred / dataset.size() << std::endl;

    }


    /*
        get cluster centroids
    */
    std::vector<std::vector<double>> get_centroids(){
        return centroids;
    }


};


}  // namespace dt