# linear_regression_calculator

A simple implementation of linear regression with a sample data file in csv format. The program reads the csv file, splits it into train and test sets, performs linear regression on the training set and tests the results on the test set. It also plots the data and regression line.

## Requirements
* numpy
* pandas
* matplotlib

## Usage

To run the program, simply run the following command:

```
python main.py
```

## Input

The program reads the `data.csv` file, which should have the following format:
- No headers
- The first line is the data
- The first column is x and the second column is y.

## Output

The program prints the estimated values and errors for each data point in the test set, as well as a plot of the data and regression line.
