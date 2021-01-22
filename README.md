# Aspect Level Sentiment Classification
## Repwalk
Run train.py file from inside `/Repwalk/` directory by providing appropriate arguments(arguments listed in train.py). Word embeddings used is Glove embedding. Extract the file [glove.840B.300d.zip](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) to `/glove/` folder inside `/Repwalk/` folder. Example command given below.
```
python3 train.py --dataset twitter
```


## PSS
Run main.py file from inside `/pss/` directory by providing appropriate arguments(arguments listed in train.py). Word embeddings used is Glove embedding. Extract the file [glove.840B.300d.zip](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) to root folder of project. Example command given below.
```
python3 main.py --batch_size 64 --dataset_name Twitter --beta .5
```
## New Model
Run train.py file from inside `/RepWalk_combined/` directory by providing appropriate arguments(arguments listed in train.py). Word embeddings used is Glove embedding. Extract the file [glove.840B.300d.zip](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) to root folder of project. Example commands are given below.
```
python3 train.py --cpt --dataset laptop --lr .001 --batch_size 32 --phi 10.0 --entropy 2.5 --eps 0.01 --beta 0.01
python3 train.py --cpt --dataset twitter --lr .001 --batch_size 64 --phi 10.0 --entropy 2.0 --eps 0.01 --beta 0.01 --num_epoch 40
python3 train.py --cpt --dataset restaurant --lr .001 --batch_size 8 --phi 10.0 --entropy 2.5 --eps 0.01 --beta 0.01
```

To run other Variants: ```--cpt``` tag is used to run with CPT and ```--no_itr``` tag is used to run without Pogressive Attention Supervision.
