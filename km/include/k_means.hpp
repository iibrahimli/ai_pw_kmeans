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

    int                               k;            // number of clusters

    std::vector<std::vector<float>>   centroids;    // centroids

    int                               max_iter;     // maximum number of iterations

    float                             delta;        // max diff of centroid between iterations

    float                             min_delta;    // stopping condition

    std::string                       init_method;  // "random" or "kmeans++"



    /*
        return index of the closest centroid to p
    */
    template <typename feat_t, std::size_t n_feat>    
    int closest_centroid(const cls_sample<feat_t, n_feat> & p){

        int   closest;
        float cl_dist = std::numeric_limits<float>::infinity();
        float tmp_dist;

        for(int cl = 0; cl < centroids.size(); ++cl){
            if((tmp_dist = euclidean_dist(p, centroids[cl])) < cl_dist){
                closest = cl;
                cl_dist = tmp_dist;
            }
        }

        return closest;
    }


public:


    /*
        k:               number of clusters
        init_method:     initialization method ("k++", "random", etc)
    */
    k_means(int k, int max_iter = 10000, float min_delta = 1e-4, std::string init_method = "kmeans++")
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

        if(dataset.size() <= k)
            throw std::runtime_error("why on earth would you cluster something like this...");

        // choose initial clusters

        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> uni(0, dataset.size());

        if(init_method == "kmeans++"){
            // k-means++
            float dist_closest = 0;
            float sum_dist     = 0;

            // first centroid - randomly chosen
            auto  c_idx      = uni(rng);
            auto  last_c_idx = c_idx;      // sentinel
            float cdf        = 0;          // cumulative distribution

            // add chosen sample to centroids
            centroids.emplace_back();
            centroids[centroids.size()-1].insert(
                centroids[centroids.size()-1].end(),
                std::begin(dataset[c_idx]._features),
                std::end(dataset[c_idx]._features)
            );

            for(int c = 0; c < k-1; ++c){

                // generate random number (used with cdf to choose centroids according to weights)
                float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

                for(c_idx = 0; c_idx < dataset.size(); ++c_idx){

                    // calculate distance to closest centroid
                    dist_closest = euclidean_dist(dataset[c_idx], centroids[closest_centroid(dataset[c_idx])]);

                    // calculate sum of distances to closest centroid for all data points


                    // choose according to D^2 probability
                    if(r < 0)
                        break;
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
        std::vector<std::size_t> assigned_clusters(dataset.size(), 0);

        for(; iter < max_iter && delta > min_delta; ++iter){

            // calculate which cluster each point belongs to
            for(int s_idx = 0; s_idx < dataset.size(); ++s_idx){
                assigned_clusters[s_idx] = closest_centroid(dataset[s_idx]);
            }

            // re-calculate centroid
        }

        return iter;
    }


    /*
        set cluster ids
        ex: k=3, [2, 0, 1]
    */
    void set_cluster_ids(std::vector<std::size_t> order){
        // std::iter_swap(begin + n, begin + k)
        for(int i = 0; i < order.size(); ++i){
            std::iter_swap(centroids.begin() + i, centroids.begin() + order[i]);
        }
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
    std::vector<std::vector<float>> get_centroids(){
        return centroids;
    }


};


}  // namespace dt