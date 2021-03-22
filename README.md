# About the project

This project demonstartes how one can leverage the power of pytorch combined with Nvidea GPU to speed up the neural network training process drastically.

For this I had used the NYC Taxi Fares dataset from kaggle. 

# About the data

The Kaggle competition provides a dataset with about 55 million records. The data contains only the pickup date & time, the latitude & longitude (GPS coordinates) of the pickup and dropoff locations, and the number of passengers. It is up to the contest participant to extract any further information. 

For this exercise I've reduced the dataset down to just 120,000 records from April 11 to April 24, 2010. The records are randomly sorted. 

# The files in the project

The folder contains the below mentioned files and subfolder:-

i. Data Folder :- The training data in the folder named Data. This file is in csv format.

ii. Data_Preprocessing.py :- This file contains the preprocessing steps. It performs 3 main steps as described below:-

STEP I. Calculate the haversine distance between two coordinates. The formula for haversine distance can be found in the     wikipedia link https://en.wikipedia.org/wiki/Haversine_formula. The distance calcuated is made a new column in the dataframe.

STEP II. Create date time columns. By creating a datetime object, we can extract information like "day of the week", "am vs. pm" etc. Note that the data was saved in UTC time. Our data falls in April of 2010 which occurred during Daylight Savings Time in New York. For that reason, I'll make an adjustment to EDT using UTC-4 (subtracting four hours).

STEP III. Separate Categorical and Continuous columns. We then separate categorical and continuous columns and convert them to tensors so that they can be fed into the pytorch model.

iii. model.py :- This file contains the logic to create a simple pytorch ANN model.

iv. model_training_cpu.ipynb :- This file demonstrates the model training using cpu. It also saves the model weights as a .pt file named TaxiFareRegrModel_cpu.pt.

v. model_training_gpu.ipynb :- This file demonstrates the model training using gpu. It also saves the model weights as a .pt file named TaxiFareRegrModel.pt.

vi. model_prediction.py :- This file can be run from the terminal to get the predicted fare using the trained model. When executed it will ask user questions like pickup latitude and longitude, dropoff latitude and longitude, number of passengers, and pickup date and time after which it gives the predicted fare.



```python

```
