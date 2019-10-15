#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "parser.hpp"
#include "dtree.hpp"

#define N_ARGS      1
#define N_FEATURES  4
#define N_GROUPS    3


using namespace std;
using namespace dt;


int main(int argc, char * argv[]){

    if(argc != N_ARGS + 1){
        cout << "Usage: " << argv[0] << " path/to/data_file.csv" << endl;
        return 1;
    }

    std::string data_filename = argv[1];


    // read and parse data

    auto t1 = std::chrono::high_resolution_clock::now();

    parser ps;
    auto [dataset, id_to_label] = ps.parse_csv<float, N_FEATURES>(data_filename);

    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    auto seconds = (int) std::round(duration / 1000000);
    auto millis  = (float) (duration % 1000000) / 1000;

    cout << "parsed " << dataset.size() << " samples in " << fixed << setprecision(2) << seconds << " s " << millis << " ms" << endl;
    cout << endl;

    // group the features
    auto ds = discretize_dataset(dataset, N_GROUPS);

    // split dataset into training and testing
    auto [train_ds, test_ds] = split_dataset(ds, 0.39, true);

    cout << "training samples: " << train_ds.size() << endl;
    cout << "test samples:     " << test_ds.size() << endl;
    cout << endl;

    // build decision tree
    decision_tree tree(2, 2);

    t1 = std::chrono::high_resolution_clock::now();

    tree.fit(train_ds);

    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    seconds = (int) std::round(duration / 1000000);
    millis  = (float) (duration % 1000000) / 1000;

    cout << "fit on " << train_ds.size() << " samples in " << fixed << setprecision(2) << seconds << " s " << millis << " ms" << endl;
    cout << endl;

    tree.evaluate(test_ds);

    return 0;
}