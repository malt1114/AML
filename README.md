
This is **Malthe Rødsgaard Pabst Lauridsen** and **Nicola Clark's** project for the course **Advanced Machine Learning for Data Science** (Spring 2024) KSAMLDS1KU



 
This project planned on exploring image segmentation across domains, by fine-tuning two models from two different domains, on the opposite domain, to see what effect this has on the models. There are a multitude of datasets available for semantic segmentation, the ones we focused on are [Cityscapes](https://www.kaggle.com/datasets/sakshaymahna/cityscapes-depth-and-segmentation) and [SUN RGB-D](https://rgbd.cs.princeton.edu/).
 

The architecture that we used for this project was a CNN encoder-decoder network with skip connections. The evaluation metric typically used for image segmentation, and therefore also utilised in this project, is Mean Intersection over Union (MIoU), which describes the average area of overlap between the predicted segmentation and the ground truth divided by their union.
 
 
## Our Main Goals: 
Pre-train a model on each dataset until the point where they can be fine-tuned.

Fine-tune the models on the opposite domains.

Evaluate the models performances on both domains to see how the performance has changed (if it has).
 
## Other Goals we explored: 
Investigate edge cases (specific scenes that underperform) in depth. Can you identify and improve the underlying problem?

Investigate latent space representation differences between the models

Compare pre-trained and fine-tuned models to generic segmentation baseline tasks

 
 
## Achieved Goals:
Train a Model on the cityscapes dataset

Evaluate the model on the cityscapes dataset

Investigate some edge cases

Learn A lot
 

 
## Preprocessing 
 
A [preprocessed version of the Cityscapes dataset from Kaggle](https://www.kaggle.com/datasets/sakshaymahna/cityscapes-depth-and-segmentation) was found and utilised for this project. 
 
The SUNGRBD data could not be located in a preprocessed format. Therefore a [pre-existing python](https://github.com/luiszeni/SUNRGBDtoolbox_python/tree/master) script from github was amended to preprocess the data and match the format to that of the cleaned Cityscapes dataset.
 
This script was altered to download an entire folder of the data, and to match the format of the Cityscapes dataset. The data that was cleaned was from the “SUNRGBD/kv2/kinect2data" folder, as the number of images lined up well with the amount in the cleaned cityscapes dataset. The re-formated data was then further cleaned as the labelling was very messy in places, i.e. many categories were mis-spelled or not grouped together (e.g. books, book, books). An [existing mapping](https://github.com/crmauceri/SUNRGBD_COCO/blob/main/seglistall.csv) of the dataset to a segmentation of 37 categories was used for this. This mapping was extended to map other common labels to a seg37 category if there was one that matched, otherwise the label was changed to "Other". An additional label of "Computer" was added during this process as there were many labels that fit this category in the dataset.
 
 
Due to size limitations on GitHub the cleaned SunRGBD data and the notebook used for cleaning can be found on the following GitHub: [SunRGBD Clean Data](https://github.com/NicolaClark/DataML)



## Architecture 


Following [an existing architecture] (https://arxiv.org/abs/1505.04597), a pytorch network was constructed and pre-trained using pytorch on the datasets to create 2 pre-trained models.



 
At this point we ran into the following issues and figured out the following fixes (not in order):

The image input was resided to a smaller size meaning the pixel values were changed by default : _Preserve the pixel values!! (KNN, not average)_
     
-1 labels never output due to ReLU in last activation layer: _Added +1 to all input labels so minimum was 0_
     
Epoch time: 10 min on CPU vs. 20-30 sec on GPU: _Utilised colab GPUs, Multithreading helped a bit too, Vectorising code, Needed to resize outputs to enable code to be run without a for loop, Using a faster data loader_


This meant that we had limited time remaining so we decided to focus on the Cityscapes model for the remainder of the project and see if we could focus on the task of semantic segmentation itself.

## Results and More Corrections 

The model on the Cityscapes dataset had a MIoU of 2.58. This was suspiciously low so we took a closer look at the outputs and realised that they were in the wrong format. We had labels that were integers and an output of floats. This is due to the original architecture needing to be adjusted from its original binary classification output layer architecture. We adjusted this by adding 20 channels (we had 20 classes) to the output layer alongside a softmax activation function. Argmax could then be used along the 20-channel-axis, in order to classify a picture’s pixels.

This architecture, with the new output layer, was then trained for 100 epochs. This would ideally have been implemented for 500 or even more epochs but due to time limitations this was not possible. This new model achieved an MIoU of 30.4. Further exploration of the corrected model including a confusion matrix and IoU per class can be found in the results notebooks.



