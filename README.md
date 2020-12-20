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
