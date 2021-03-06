# Safe Seizure

Safe Seizure is a machine/deep learning model aiming to predict brain seizures based on intracranial EEGs (Electro Encephalograms) from epileptic human patients.
Being able to predict brain seizures may give epileptic patients enough time to safely anticipate their next seizure, potentially avoiding life-threatening consequences.

Link to presentation -> https://www.canva.com/design/DAEiSbS_shA/c0jf8B3lhZuVd2dsCCBNOA/view?utm_content=DAEiSbS_shA&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink


# Goal of project

The first objective of this project is to accurately classify EEG sequences as being interictal OR preictal, therefore determining whether a seizure will occur within the next 5 to 65 mins.

An additional objective would be to identify how close to a seizure a patient is by narrowing down the anticipation time range.

# Data

The data is sourced from the American Epilepsy Society Seizure Prediction Challenge on Kaggle: https://www.kaggle.com/c/seizure-prediction/data.
The raw data consists of two folders Patient_1 and Patient_2 each containing ~260 EEGs in .mat format (MatLab formatted files).

Each folder contains three types of EEGs:
* Interictal EEGs, corresponding to a non_seizure signal
* Preictal EEGs, corresponding to a pre-seizure signal (recorded from 65 minutes before a seizure occurred, see image below)
* Test EEGs, corresponding to the unlabelled test set of either preictal or interictal sequences.

<p align="center">
<img src="images/kaggle_data.png" width="600" height="350"/>
<p/>

<p align="center">Blue signal: Preictal sequence; Red signal: Ictal sequence (actual seizure)</p>
<p align="center">Image credits: American Epilepsy Society Seizure Prediction Challenge</p>

# Methodologies

## Machine Learning

The first methodology used is machine-leanring based, following Al-Qerem et al. 2020 (https://arxiv.org/pdf/2102.01647.pdf)

(1) Independent Component Analysis (using `FastICA`)

(2) Discrete Wavelet Transform (using `pywvt`)

(3) Feature extraction: `mean`, `average power`, `mean absolute value`, `Shannon entropy`

(4) Regression

## Deep Learning

(1) Low- and high-pass filtering (60 Hz and 0.5 Hz respectively)

(2) Convert time sequences to spectrograms (using Fast Fourier Transform)

(3) CNN classification of spectrograms

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for SafeSeizure in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/SafeSeizure`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "SafeSeizure"
git remote add origin git@github.com:{group}/SafeSeizure.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
SafeSeizure-run
```

# Install

Go to `https://github.com/{group}/SafeSeizure` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/SafeSeizure.git
cd SafeSeizure
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
SafeSeizure-run
```
