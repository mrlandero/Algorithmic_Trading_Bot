# Algorithmic_Trading_Bot
Combining algorithmic trading with financial Python programming and machine learning to create an algorithmic trading bot that learns and adapts to new data and evolving markets.

--

## Technologies

This project leverages python 3.7 with the following packages:

**[Pandas Library](https://pandas.pydata.org/)** - pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,
built on top of the Python programming language.<br>

**[Numpy Library](https://numpy.org/)** - NumPy offers comprehensive mathematical functions, random number generators, linear algebra routines, Fourier transforms, and more.<br>

**[Pathlib Library](https://pathlib.readthedocs.io/en/pep428/)** - This module offers a set of classes featuring all the common operations on paths in an easy, object-oriented way.<br>

**[HvPlot Library](https://hvplot.holoviz.org/)** - A high-level plotting API for the PyData ecosystem built on HoloViews.<br>

**[MatPlotLib Library](https://matplotlib.org/)** - Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.<br>

**[SkLearn SVM Library](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)** - C-Support Vector Classification.<br>

**[SkLearn.preprocessing Standard Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)** - Standardize features by removing the mean and scaling to unit variance.<br>

**[Pandas DateOffset Library](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.tseries.offsets.DateOffset.html)** - Standard kind of date increment used for a date range.<br>

**[SkLearn Classification Report Library](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)** - Build a text report showing the main classification metrics.<br>

**[SkLearn LogisticRegression Library](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)** - Logistic Regression (aka logit, MaxEnt) classifier.<br>

--

## Usage

To use the Venture Capital Neural Network application, simply clone the repository and run the Jupyter Notebook **machine_learning_trading_bot.ipynb** either in VSC, or in Jupyter Lab.

Step 1: Imports

```python
# Imports
import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
```

Step 2: Import the OHLCV dataset into a Pandas DataFrame.

The following image displays the output of this step:

![Import the Data to Create DataFrame](read_csv.png)

Step 3: Filter the date index and close columns and use the pct_change function to generate  returns from close prices.

The following image displays the output of this step:

![Filter the DF and Add Column](filtered_df.png)

Step 4: Set the short window and long window and generate the fast and slow simple moving averages (4 and 100 days, respectively).

The following image displays the output of this step:

![Fast and Slow Simple Moving Averages](fast_slow_sma.png)

Step 5: # Initialize the new Signal column. When Actual Returns are greater than or equal to 0, generate signal to buy stock long. When Actual Returns are less than 0, generate signal to sell stock short.

The following image displays the output of this step:

![Signal Column](signal_column.png)

Step 6: Calculate the strategy returns and add them to the signals_df DataFrame.

The following image displays the output of this step:

![Strategy Returns](strategy_returns.png)

Step 7: Plot Strategy Returns to examine performance.

The following image displays the output of this step:

![Strategy Returns Plot](strategy_returns_plot.png)

Step 8: Assign a copy of the sma_fast and sma_slow columns to a features DataFrame called X.

The following image displays the output of this step:

![Create X DataFrame](features_df.png)

Step 9: Create the target set selecting the Signal column and assiging it to y.

The following image displays the output of this step:

![Target Set](target_set.png)

Step 10: Select the start of the training period.

The following image displays the output of this step:

![Training Start](training_start.png)

Step 11: Select the ending period for the training data with an offset of 3 months.

The following image displays the output of this step:

![Training End](training_end.png)

Step 12: Generate the X_train and y_train DataFrames.

The following image displays the output of this step:

![X and y Training Sets](x_y_training_sets.png)