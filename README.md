# stl-10_kmeans

Data source: [`train.t7b`](https://s3.amazonaws.com/dsga1008-spring16/data/a2/train.t7b), [`val.t7b`](https://s3.amazonaws.com/dsga1008-spring16/data/a2/val.t7b), [`extra.t7b`](https://s3.amazonaws.com/dsga1008-spring16/data/a2/extra.t7b), [`test.t7b`](https://s3.amazonaws.com/dsga1008-spring16/data/a2/test.t7b).

Platform: [torch](http://torch.ch/).

Requirement: torch, [cuda-toolkit](https://developer.nvidia.com/cuda-toolkit).

## Abstract
The data is part of image process dataset [stl-10](https://cs.stanford.edu/~acoates/stl10/). We have 4000 train data, 1000 validation data, 8000 test data, 100000 extra unlabeled data. So it is a semi-supervised problem. To use the unlabel data, we use k-means to get the initial kernel for the first four layers of CNN, and then start train the whole network. The reason why it takes effect is because the problem is not a convex problem, therefore a bad initial value will lead to local minimum. I would like to say some intreseting part of this program.

### Kernel
Because the kernel size is 3x3, we don't need the whole picture (96x96). What we did here is to find 10 (3x3) pictures where it has great gradient for each input (96x96) and then randomly select one. It is in `find_patch.lua`. Considering the memory of GPU, we will randomly select 10000 unlabeled data. Then we use kmeans to find the first layer kernel.

To find the next layer kernel, just get the output of the fourth layers (CNN->BatchNormalization->RELU->MaxPooling), then use the same method to find the kernel for the fifth layers. All of this is in `find_patch.lua`.

### Augmentation
We think that it is not enough for the train data. So we use some augmentation technology to generate more train data.

1. Scaling
2. Translation
3. Rotation

However, it performs bad if the most part of train data is the augmentation data. So we just generate 2 times for each train data. And this is in `augmentation.lua`.

## How to run
To just get the prediction, you can simply run `result.lua`.

But to test the other code, you must run `provider.lua` first to get the data.