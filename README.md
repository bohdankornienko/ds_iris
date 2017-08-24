# Iris classification
Solution for classification task with Iris Data set

# Dataset
Dataset can be downloaded from page: http://archive.ics.uci.edu/ml/datasets/Iris

# Results
Several experiments were performed. Mainly exploring layer size and relu usage on prediction result.

## Experiment with layer sizes in range 1-25
In a given layers range the best resulst can be obtained with layer size 20

with learning rate 0.005, with relu, epoch count 100
accuracy is 0.9666666666666667

with learning rate 0.005, without relu, epoch count 500
accuracy is 0.9666666666666667

## Final resuls
Making layer size up to 1024 and sitch to 500 epoch with relu
And it gives accuracy 1.0

There is a set of values wich also give the same results. The set is following:
314, 315, 320, 350, 410, 512

Final layer size for model: 512

Using tf randome seed 1 converges model to accuracy 1.0

# Further steps to do
* Reproduce Fisher's steps from his paper
* Visualize dataset with TSNE

