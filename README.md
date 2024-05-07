# CS598-DL4H-Project
Course project on CS598 - Deep Learning for Healthcare  
Selected research paper: [TransfromEHR](https://www.nature.com/articles/s41467-023-43715-z), [Github](https://github.com/whaleloops/TransformEHR)  
Video [link]()  
Group 91: Abilash Bodapati, Sneha Sarpotdar, Sotheara Chea


## Google Collab setup
* Download iPython notebook. *[instruction](https://colab.research.google.com/github/quantumlib/Cirq/blob/master/docs/tutorials/google/colab.ipynb#scrollTo=6q5lpwgW5TrE)* 
* Upload notebook to Google Colab. *[instruction](https://colab.research.google.com/github/quantumlib/Cirq/blob/master/docs/tutorials/google/colab.ipynb#scrollTo=czt1fSEHooF9)*  
<img src="docs/image.png" width="400">


##  Environment
#### Operating systems:
* Ubuntu 20.04.5 LTS
* GPU T4
* Google colab environment
* Python 3.8.11 with libraries:
* NumPy (currently tested on version 1.20.3)
* PyTorch (currently tested on version 1.9.0+cu111)
* Transformers (currently tested on version 4.16.2)
* tqdm==4.62.2
* scikit-learn==0.24.2


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

