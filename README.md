# Captcha_Classification
MSBD5012 Machine Learning Project

-----
Environment:
- Programs are run on linux environment, with a 6-core GPU

Required packages to install
- pytorch 1.10.0
- cleverhans 4.0.0
- opencv 4.4.0
- numpy 1.17.0
- matplotlib 3.3.4
- tqdm 4.62.3

------
## Preparation
Download the train.zip and test.zip from <a href="https://drive.google.com/drive/folders/1clyJvuedGKL7mWSatx1pCnOFKUWFJzBo?usp=sharing">the download link</a>, unzip the two files and put under the data folder as `/data/train` and `/data/test`

## Run the program
Run the following command using bash.
```
bash ./run.sh
```
- It will train 5 classification models and save the model under model folder
- Then it will generate the adversarial images using 5 attack methods and save under path `/data/attack_image`
- Then it will evaluate the results and generate a `result.csv` and `RMSD.csv` files.

## Report
The project report can be found [here](assets/Adversarial_attack_on_captcha.pdf)

## Presentation
Our project presentation can be found in [Youtube](https://www.youtube.com/watch?v=-txlUtuKonA&list=LL&index=1&t=1s).
