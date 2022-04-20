# VTC-preparation
In order to prepare the dataset for voice type diarization, you need to split it into train, validation and test subsets and then generate annotations.
Your dataset should be of the standardized ChilProject format (https://childproject.readthedocs.io/en/latest/) and be stored in a Datalad repository (https://www.datalad.org/). 
The directory where you are going to create your dataset should be a YODA directory (https://handbook.datalad.org/en/latest/basics/101-127-yoda.html).

First, create a remote Datalad directory of your corpus:
`datalad install [address of Datalad repository]`

`datalad get [content of your Datalad repository]`

Then you can launch the train-dev-test split. Indicate a) the path to remote Datalad directory; b) the path to YODA input directory; c) provide a file.txt with a given split of data between train, validation and test subsets, edited according to the given example (optional).

`python train-dev-test.py --corpus [path to Datalad remote with your corpus] --output [path to YODA/input] --dict_children file.txt`

