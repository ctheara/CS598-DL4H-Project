# CS598-DL4H-Project
Course project on CS598 - Deep Learning for Healthcare  
Selected research paper: [TransfromEHR](https://www.nature.com/articles/s41467-023-43715-z), [Github](https://github.com/whaleloops/TransformEHR)  
Video [link]()  
Group 91: Abilash Bodapati, Sneha Sarpotdar, Sotheara Chea


## Google Collab setup
* Download iPython notebook. *[instruction](https://colab.research.google.com/github/quantumlib/Cirq/blob/master/docs/tutorials/google/colab.ipynb#scrollTo=6q5lpwgW5TrE)* 
* Upload notebook to Google Colab. *[instruction](https://colab.research.google.com/github/quantumlib/Cirq/blob/master/docs/tutorials/google/colab.ipynb#scrollTo=czt1fSEHooF9)*  
<img src="docs/image.png" width="400">



## Data Setup
#### To get the Dataset mounted on Google Drive:  

*  Go to [MIMIC IV Website](https://physionet.org/content/mimiciv/2.2)
*  Download the dataset to your google drive
*  Validate the dataset is present under -> `MyDrive/mimiciv/2.2/hosp`  
    Example:
    *   `/content/drive/MyDrive/mimiciv/2.2/hosp/admissions.csv.gz`
    *   `/content/drive/MyDrive/mimiciv/2.2/hosp/diagnoses_icd.csv.gz`

#### Mount Notebook to Google Drive
```
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
````


## Imports Modules
```sh
!pip install --upgrade accelerate
!pip install --upgrade transformers
!pip install --upgrade tqdm
!pip install --upgrade scikit-learn
!pip install --upgrade datasets
```
```python
from google.colab import drive
import pandas as pd
from datetime import datetime
import random
import math
import numpy as np
import pandas as pd
from google.colab import drive
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple
import csv
import json, time
from collections import defaultdict
from itertools import combinations, islice
import pickle

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

import torch
torch.__version__
import torch.nn.functional as F
from torch import Tensor, nn

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler

from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput
from transformers.training_args import TrainingArguments

from transformers.activations import ACT2FN
from transformers.models.bart.configuration_bart import BartConfig
from transformers import BertTokenizer, BartTokenizer, BartForConditionalGeneration, Trainer, TFTrainingArguments
from transformers import DefaultDataCollator

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    TrainingArguments,
    set_seed,
)

from dataclasses import dataclass, field

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
```