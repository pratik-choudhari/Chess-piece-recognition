# Chess-piece-recognition
A project on chess piece identification from images.

![](https://img.shields.io/badge/dependency-Augmentor-brightgreen)

Dataset credits: [DatasetDaily](https://github.com/Dataset-Daily)

__Running this code on 8GB or less RAM device might crash the system__

Before executing the code:
- `pip3 install -r requirements.txt`
- Download data from [here](https://www.kaggle.com/niteshfre/chessman-image-dataset) and place `Chess` directory in `data` folder
- Run `utils.py`, this will:
  - Segment data into train, validation, test folder in 70,10,20 portions
  - Augment train images and store in `train/output`
