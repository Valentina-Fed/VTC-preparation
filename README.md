# VTC-preparation
In order to prepare the dataset for voice type diarization, you need to split it into train, validation and test subsets and then generate annotations.
Your dataset should be of the standardized ChilProject format (https://childproject.readthedocs.io/en/latest/) and be stored in a Datalad repository (https://www.datalad.org/). 
The directory where you are going to create your dataset should be a YODA directory (https://handbook.datalad.org/en/latest/basics/101-127-yoda.html).

First, create a remote Datalad directory of your corpus:

`datalad install [address of Datalad repository]`

`datalad get [content of your Datalad repository]`

Then you can launch the train-dev-test split. First, create a YODA directory with a Datalad subdirectories "input", and in the "input" create subdirectories "train", "dev" and "test".

Indicate a) the path to remote Datalad directory; b) the path to YODA input directory; c) provide a split.txt with a given split of data between train, validation and test subsets, edited according to the given example (optional).

`python train-dev-test.py --corpus [path to Datalad remote with your corpus] --output [path to YODA/input] --dict_children example_split.txt`

In order to create annotations rttm and uem, used for VTC training (see examples here: https://github.com/MarvinLvn/pyannote-audio/tree/be57f5ef4c01c79b087b6f817979e7577309e797/tutorials/data_preparation), launch the following command:

`python create_rttm.py --corpus [path to Datalad remote with your corpus] --output [path to YODA/input]`

The script parse ChildProject annotations, but there is an option also to parse raw TextGrid annotations.
It will create subdirectories "gold" in the subdirectories train, dev and test with rttm and uem annotations for each audio-file of your corpus, and two files with annotations for the whole corpus: rttm and uem.
