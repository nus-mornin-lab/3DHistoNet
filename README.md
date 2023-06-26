# Data-Efficient Computational Pathology Platform for Faster and Cheaper Breast Cancer Subtype Identifications

### Authors
*  Kideog Bae (co-first) (anthony.bae92@gmail.com)
* Young Seok Jeon (co-first) (youngseokejeon74@gmail.com)
* Yul Hwangbo (yulhwangbo@ncc.re.kr)
* Chong Woo Yoo
* Nayoung Han (co-coress) (hanny@ncc.re.kr)
* Mengling Feng (co-coress) (ephmf@nus.edu.sg)
---
## Abstract
>**IMPORTANCE**
Breast cancer subtyping is a crucial step in determining therapeutic options, but the molecular examination based on immunohistochemical (IHC) staining is expensive and time-consuming. Deep learning model opens up the possibility to predict the subtypes based on the morphological information from hematoxylin and eosin (H&E) staining, a much cheaper and faster alternative. However, training such predictive models conventionally requires a massive number of histology images that leads to high costs and used to be infeasible for single institute . 
**OBJECTIVE**
To develop a data-efficient computational pathology platform, 3DHistoNet, that is capable to learn from a relatively small number of histology images for accurate predictions of breast cancer subtypes.
**DESIGN, SETTING, AND PARTICIPANTS**
We retrospectively examined 420 cases of primary breast carcinoma patients diagnosed between 2018 and 2020 at the Department of Pathology, National Cancer Center (South Korea) pathology slides of the breast carcinoma patients were prepared according to the standard protocols. Age, gender, histologic grade and hormone receptor (ER, PR) status/HER2 status/Ki-67 index were evaluated by reviewing medical charts and pathological records. 
**MAIN OUTCOMES AND MEASURES**
The area under the receiver operating characteristic curve (AUC) and decision curve were analyzed to evaluate the performance of our 3DhistoNet platform for predicting the ER, PR, AR, HER2 and Ki67 subtype biomarkers with a 5-fold cross validation.
**RESULTS**
We demonstrate 3DHistoNet can predict all clinically important biomarkers (ER, PR, AR, HER2 and Ki67) with performance exceeding the conventional multiple-instance-learning models by a considerable margin (AUC: 0.75-0.91 vs 0.67-0.8). We further show that our novel histology scanning approach can make up the limitation of insufficient training dataset without any additional cost. Finally, 3DHistoNet offers an additional capability to generate attention maps which reveal correlations between histologic features and biomarker expressions.
**CONCLUSIONS AND RELEVANCE**
Our platform, with its high prediction capability and versatility, is an appealing tool as an effective prediction tool for the breast cancer subtype biomarkers. Its development would encourage morphology-based diagnosis which is faster and less error-prone compared to the protein quantification method based on IHC staining.
---
## To Run
###  1. Data

The data we use in the paper is not publictly available. Instead, we provide pre-processing codes that users can use to preprocess their own 3D histology dataset.

* Given a folder with patient level 3D histology images in **.tif** format, the first step that users are required to perform is 1. standardizing the color distribution across the images and 2.chopping them into multiple patches. The two steps are done in a single script named **preprecess.py** and the generated patches are store in **.h5** format. User are required to modify the script to specify the directory of the raw **.tif** images and the directory of the processed **.h5** images.

* The next pre-processing step is to turn each **.h5** to **.npy** images for faster I/O during SSL pre-training. This is done by a script named **flatten.py**. To run the script, users are required to modify the script to specify the directory of the **.h5** images and the directory where the processsed **.npy** will be stored.

### 2. SSL Pretrain & Feature extraction

As described in the abstract, our proposed 3D histology diagnosis model is comprised of two steps: 1) SSL pretraining of a feature extractor to obtain semantic features from the cropped 3D histology images, and 2) prediction model that takes the extracted semantic features as its input.

This subsection covers the first step: 1) how to SSL pretrain the feature extractor, and 2) using the pretained feature extractor to obtain semantic features and store them as files.

* To SSL pretrain the feature extractor run the command below. The scripts uses ResNet50 as the default encoder. Hence, user are required to modify the script if want to test on other encoders types. 
* The SSL pretraining will produce checkpoints for every 10th epoch.
```sh
pretrain.py --tecnique SIMCLR --datapath path/to/npy/files --ckptPath path/to/store/ckpt
```
 * To extract features using one of the checkpoints produced from the previous command, run the command below. Users are required to specify **dataLoadPath**, **ckptLoadPath**, and **dataSavePath**.
 
```sh
extractFeatures.py --dataLoadPath path/to/h5data --dataSavePath path/to/store/feature --ckptLoadPath path/to/ckpt
```

### 3. Prediction model training

This subsection cover the second step of our proposed model: training a 1D CNN prediction model. This step requires a **csv file that contains the subject-id and its cancer subtype labels**. Below is the command to train the prediction model. Users need to specify **diagnosis type**, **MIL method**, **SSL type**, and **data type**. For example, to train on ER subtype, attention MIL, SIMCLR SSL pretrain, and 3D data, run

```sh
./train.sh ER attention SIMCLR 3D 0
```