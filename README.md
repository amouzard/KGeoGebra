
# Knowledge GeoGebra: Leveraging Geometry of Relation Embeddings in Knowledge Graph Completion
**Introduction**

This is the PyTorch implementation of the [KGeoGebra](https://...) model for knowledge graph embedding (KGE). 
For more information, please refer to the following GitHub page
[GraphVite](https://github.com/DeepGraphLearning/graphvite).

**Implemented features**

Models:
- [x] Ellipse
- [x] ELiipsEs
- [x] ButtErfly
- [x] ButtErflies
 - [x] RotatE
 - [x] TransE
 - [x] ComplEx
 - [x] DistMult

Evaluation Metrics:

 - [x] MRR, MR, HITS@1, HITS@3, HITS@10 (filtered)
 - [x] AUC-PR (for Countries data sets)

Loss Function:

 - [x] Uniform Negative Sampling
 - [x] Self-Adversarial Negative Sampling

**Usage**

Knowledge Graph Data:
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE models are evaluated on this data set
- *X_test.txt*: the KGE models are also evaluated on 9 pattern-specific test data sets. X stands for A, C, S, AC, AS, CS, ACS, UA, UC, and US.

**Train**

For example, this command train the EllipsE model on FB15k-237 dataset with GPU 0.
```
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/FB15k-237 \
 --model EllipsE \
 -n 256 -b 256 -d 800 \
 -g 6.0 -a 1.0 \
 -lr 0.0001 --max_steps 150000 \
 -save models/EllipsE_FB15k-237_0 --test_batch_size 16 -de
```
   Check argparse configuration at codes/run.py for more arguments and more details.

**Test**

    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

The sub test datasets are saved in a folder named Test. When the argument --do_sub_test is switched 
on, the models are evaluated on the sub test datasets. 

**Reproducing the best results**

The run.sh script provides an easy way to search hyper-parameters:

    bash run.sh train EllipsE FB15k-237 0 0 256 256 800 6.0 1.0 0.0001 150000 16 -de

**Using the library**

The python libarary is organized around 3 objects:

 - TrainDataset (dataloader.py): prepare data stream for training
 - TestDataSet (dataloader.py): prepare data stream for evaluation
 - KGEModel (model.py): calculate triple score and provide train/test API

The run.py file contains the main function, which parses arguments, reads data, initilize the model and provides the training loop.

