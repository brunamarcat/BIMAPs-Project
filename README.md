# Few- Shot Learning: How many labels do we need to achieve a good performance on unknown data?

This project uses DeepD3, which can be found in:

- DeepD3 [repository](https://github.com/ankilab/DeepD3/tree/main)
- DeepD3 [website ](https://deepd3.forschung.fau.de/)

Project developed by **Bruna Martin i Català**

## Description

This project is part of the Biomedical Imaging Project course at FAU and aims to explore how can few-shot learning be applied to DeepD3 and how does this perform when using less data to train the model. The chosen few-shot method used to do so, is a contrastive loss method called triplet loss using a Siamese Network. 

A Siamese Network often consists of two or more networks that share weights and are "physically connected", thus their name. Differently from traditional networks that have one input and one output, the Siamese network used in this project has three input branches and one output branch. The three inputs will be three images called "anchor", a randomly chosen image; "positive", a copy of the anchor image but with applied data augmentation;  and "negative" a different image from the anchor from the dataset. 

This triplet loss measures how closely the network's output for a positive pixel example resembles another instance of the same pixel label, while simultaneously measuring its dissimilarity to examples from different pixel labels (negative examples).  The goal is to minimize the distance between positive pairs and maximize the distance between negative pairs and use this information as an auxiliary loss to train the model. 

## Usage

How to use the explorative code:

The execution code the parameters to be used need to be defined. Here is the list of the parameters:

| Argument            | Description                                                  | Default Value                |
| ------------------- | ------------------------------------------------------------ | ---------------------------- |
| `generator_data`    | Kind of data generator to use: 'DataGeneratorStreamSiamese' or 'DataGeneratorStream'. | 'DataGeneratorStreamSiamese' |
| `samples_per_epoch` | Number of samples per epoch                                  | 5000                         |
| `batchsize`         | Batch size                                                   | 32                           |
| `ntripletsperimage` | Number of triplets per image for TripletLoss                 | 1                            |
| `EPOCHS`            | Number of epochs for training                                | 100                          |
| `loss1`             | Which loss is used: DiceLoss or TripletLoss                  | 'TripletLoss'                |
| `loss2`             | Which loss is used: MSE or TripletLoss                       | 'TripletLoss'                |
| `margin1`           | Margin for the loss1 function when TripletLoss is used       | 0.5                          |
| `margin2`           | Margin for the loss2 function when TripletLoss is used       | 0.5                          |
| `l1`                | Lambda for the loss1 function when TripletLoss is used       | 0.7                          |
| `l2`                | Lambda for the loss2 function when TripletLoss is used       | 0.7                          |
| `lr`                | Initial learning rate for optimizer                          | 0.001                        |
| `epoch_decay`       | When to start to decay the learning rate                     | 15                           |
| `value_decay`       | Value for exponential decay of the learning rate             | 0.025                        |
| `number_runs`       | Number of runs of the experiment to be done                  | 1                            |
| `data_path`         | Path to the data                                             | "./"                         |
| `data_results`      | Path where the results will be stored                        | "./"                         |
| `augmentation`      | Augmentation or not of the data                              | False                        |

An example of how to execute the code to train the model using the Triplet Loss would be using the following lines:

python 'Training DeepD3 model.py'  \

  `--GeneratorData DataGeneratorStreamSiamese \`

  `--samples_per_epoch 5000 \`

  `--ntripletsperimage 1\`

  `--epochs 100 \`

  `--batchsize 32 \`

  `--loss1 TripletLoss \`

  `--l1 1.4 \`

  `--margin1 0.5 \`

  `--loss2 TripletLoss \`

  `--l2 0.7 \`

  `--margin2 0.5 \`

  `--lr 0.001 \`

  `--epochdecay 15 \`

  `--valuedecay 0.025 \`

  `--number_runs 1 \`

  `--data_path ./ \`

  `--data_results ./ \`

  `--augmentation False`

## Results

Here the results obtained training the model using 5000 examples (10% of the data)  using the original dice loss and mean squared error in contrast to training it with 5000 examples using the triplet loss method are shown. The parameters used are the ones written in the example above.
![Sample results  spines](https://github.com/user-attachments/assets/8e94964d-ffb8-43db-80ba-56e76cc43472)
![Sample results dendrites](https://github.com/user-attachments/assets/90a04882-cf86-4ee0-aab2-cd5905799619)
