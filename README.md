# BAST

[BAST: Binaural Audio Spectrogram Transformer for Binaural Sound Localization]().
---
## Architecture

![BAST](https://github.com/ShengKuangCN/BAST/blob/main/figure/01.BAST_architecture.pdf)

## Prerequisites

  pip install -r requirements.txt

## Datasets

For the audio dataset we use, please contact s.mehrkanoon@uu.nl or siamak.mehrkanoon@maastrichtuniversity.nl for further information.

After downloading the dataset, please unzip files into "\data" directory. The traning and testing dataset should be under "\data\train" and '\data\test' respectively.

## Usages

### Train
We offer several training options as below
* For binaural integration method (--integ): 
   * 'ADD': Addition 
   * 'SUB': Subtraction
   * 'CONCAT': Concatetation
* For loss Function (--loss): 
   * 'MIX': Hybrid loss
   * 'MSE': Mean Square Error(MSE) loss
   * 'AD': Angular Distance(AD) loss
* For weights sharing (--shareweights)
* For using differentr auditory environments (--env): 
   * 'RI01': anechoic environment
   * 'RI02': lecture hall (reverberation environment)
   * 'RI': anechoic environment + lecture hall

#### example
For training BAST with subtraction integration and hybrid loss:
    
    python train_BAST.py --integ SUB --loss MIX --env RI

For training BAST with subtraction integration, hybrid loss and weights-sharing:
    
    python train_BAST.py --integ SUB --loss MIX --shareweights --env RI

### Test

For testing BAST in the anechoic environment + lecture hall:
    
    python eval_BAST.py --integ SUB --loss MIX --env RI

For testing BAST in the lecture hall:

    python eval_BAST.py --integ SUB --loss MIX --env RI02

## Performance

Model | Loss | Integ. | AD | MSE
:---: | :---: | :---: | :---: | :---:
BAST-NSP  | MSE | Concat. | 2.78° | 0.003
BAST-NSP  | MSE | Add. | 2.48° | 0.002
BAST-NSP  | MSE | Sub. | 2.42° | 0.002
BAST-NSP  | AD | Concat. |2.39° | 
BAST-NSP  | AD | Add. |1.30° | 
BAST-NSP  | AD | Sub. |1.63° | 
BAST-NSP  | Hybrid | Concat. |2.76° | 0.004
BAST-NSP  | Hybrid | Add. |1.83° | 0.002
BAST-NSP  | Hybrid | Sub. |1.29° | 0.001
BAST-SP  | MSE | Concat. |2.02° | 0.002
BAST-SP  | MSE | Add. |4.97° | 0.018
BAST-SP  | MSE | Sub. |1.94° | 0.002
BAST-SP  | AD | Concat. |2.66° | 
BAST-SP  | AD | Add. |13.87° | 
BAST-SP  | AD | Sub. |1.43° | 
BAST-SP  | Hybrid | Concat. |1.98° | 0.003
BAST-SP  | Hybrid | Add. |5.72° | 0.026
BAST-SP  | Hybrid | Sub. |2.03° | 0.002

## Citation

