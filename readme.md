# Car Break-in Detector

![car](https://github.com/ian-andriot/CarBreakinDetector/blob/main/images/car/0013.jpg) ![breakin](https://github.com/ian-andriot/CarBreakinDetector/blob/main/images/glass/68.jpeg)

## Table of Contents

- [Problem Statement](#Problem-Statement)
- [Summary](#Summary)
- [Results](#Results)
- [Demo](#Demo)
- [Opportunities for Further Development](#Opportunities-for-Further-Development)

## Problem Statement

At most, a car is only protecting it's contents with a sheet of glass. If a thief saw something that they liked, they could simply smash the window, grab it, and run. Is it possible to make a detector that could sound an alarm as this is happening using data science?

## Summary

The first step to any data project is acquiring said data. Data from [kaggle](https://www.kaggle.com/anujms/car-damage-detection) and [peltarion](https://peltarion.com/knowledge-center/documentation/tutorials/car-damage-assessment) were the only places with prebundled images of car damage found to start the project. Between both of these datasets, there was only about 200 images depicting a car with broken glass. For this reason the [flickr](https://www.flickr.com/) api was used in [this notebook](https://github.com/ian-andriot/CarBreakinDetector/blob/main/1_flickr_api.ipynb) to acquire additional images. The target was also expanded to include any broken glass. This totalled to 618 images for the broken glass target and 1150 images for cars - the car class using the kaggle dataset exclusively.

After acquiring data, exploratory data analysis was then conducted in the [eda notebook](https://github.com/ian-andriot/CarBreakinDetector/blob/main/2_eda.ipynb). Preprocessing techniques such as scaling, resized, and Histogram of Oriented Gradients were also explored here.

The preprocessing techniques previous noted were then applied in the [modeling notebook](https://github.com/ian-andriot/CarBreakinDetector/blob/main/3_modeling.ipynb). Modeling techniques utilizing Support Vector Machines and Convolutional Neural Networks were used as well Bayesian Optimization and Hyperbanding for hyperparameter tuning.

## Results

The final model ([Saved Models/tuned_cnn.tflite](https://github.com/ian-andriot/CarBreakinDetector/blob/main/Saved%20Models/tuned_cnn.tflite)) results in an accuracy score of 96% and a roc auc score of 0.98 on the validation set. The most common images that are missed seems to be when a window is smashed cleaned with little glass left or if there is too much added augmentation.

## Demo

To top the project off a [demo](https://mighty-garden-08758.herokuapp.com/) was deployed to heroku that was based off of streamlit. The demo sourcecode is located [here](https://github.com/ian-andriot/CarBreakinDetector/tree/main/streamlit). The images in this demo are pre grayscaled and resized to 128x128.

For this demo, users can adjust the level of image augmentation to generate a batch of 32 images from the dataset. They can then predict whether or not a break-in has occured using the final model. When predicting a batch the demo will sort between four columns: true positives, true negatives, false postives, and false negatives. The goal of this demo was to help visualize what the final model is capable of in a variety of conditions.

## Opportunities for Further Development

- While the resulting model appears to be good, will this model only work well while viewing an individual car or can it be scaled to include a parking lot?
- A larger dataset to account for more situations.
- Processing a livecam feed to trigger an alert.
- An API that processes requests using this model.
- Incorporating a model that uses sound as an input to detect breaking glass.