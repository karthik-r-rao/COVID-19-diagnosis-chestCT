# COVID-19-diagnosis-chestCT

**An attempt at building a classifier which can detect the presence of COVID-19 from 2-D slices of Computerized Tomography (CT) scans. This project was fully implemented using the PyTorch framework.**

The dataset was obtained from [this repo](https://github.com/UCSD-AI4H/COVID-CT). There is a suggested split in the same repo. [This](https://drive.google.com/drive/folders/1TS6AKegWTRJoXOZlixhiAPencnCWQn1_?usp=sharing) folder has the dataset which was obtained after following the suggested data-split procedures. (Read the *COVID-CT-MetaInfo.xlsx* file for dealing with the raw data)

The models were built looking at architectures like VGG-16 and ResNet. I experimented with the architectures and how it impacts performance. Introducing feedforward connections generally helps improve the performance of your model. That was one of the highlights of the ResNet paper. Here are the links to the papers-

* [VGG-16](https://arxiv.org/pdf/1409.1556.pdf)
* [ResNet](https://arxiv.org/pdf/1512.03385.pdf)

For *visualizing* a model's performance, a technique called GradCAM can be used. This can answer the 'why' behind the model's decision. GradCAM was implemented from scratch, as PyTorch doesn't seem to have an already existing function. Here is the [paper](https://arxiv.org/pdf/1610.02391.pdf) on GradCAM. 
