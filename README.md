# Sentiment-Analysis

 > Classification of sentiment140 dataset as positive and negative using LSTM

## Dataset
This is the [sentiment140](http://help.sentiment140.com/for-students) dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

## Preprocessing
For data preprocessing, the following steps are done in order:
- Convert text to lowercase
- Remove User Handle
- Remove http links
- Remove Digit
- Remove Space
- Stemming words with [NLTK](https://www.nltk.org/)
- Lemmatize Words with [NLTK](https://www.nltk.org/)
- Remove Stop Words

## Word Embeddings
[GloVe embeddings](https://nlp.stanford.edu/projects/glove/) with embedding_dim=300 is used for vector representations for words.

## Net
The network uses an [Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) layer, two bidirectional [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) layers and a [fully connected(linear)](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear) layer for calculating the output probability.

## Results
| Train Acc.      | Validation Acc. | Test Acc.  | Test loss. |
| :-------------: | :-------------: | :--------: | :--------: |
|      0.77       |      0.76       |    0.76    |    0.49


## Setup & instructions
1. Open Anaconda Prompt and navigate to the directory of this repo by using: ```cd PATH_TO_THIS_REPO ```
2. Execute ``` conda env create -f environment.yml ``` This will set up an environment with all necessary dependencies.
3. Activate previously created environment by executing: ``` conda activate sentiment-analysis ```
4. Training and/or testing the model.

    a) Start the run script: ``` python src/run.py ``` which will automatically instantiate the model and start training it after dataset is loaded. After training the model performance will be evaluated on the test set.

