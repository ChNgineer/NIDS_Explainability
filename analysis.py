import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

def histogram(data, bins=10, title="Histogram", xlabel="Value", ylabel="Frequency"):
    plt.hist(data, bins=bins, edgecolor='black')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()
    
def multi_histogram(df, n, m, bins=20, title="Histograms"):
    """
    Plot n by m histograms from a pandas dataframe.

    Parameters:
    - df: pandas df with n * m columns of data
    - n: number of rows of histograms
    - m: number of columns of histograms
    """

    fig, axes = plt.subplots(n, m, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)

    for i in range(n):
        for j in range(m):
            col_index = i * m + j
            axes[i, j].hist(df.iloc[:, col_index], bins=bins, color='skyblue', edgecolor='black')
            axes[i, j].set_title(df.columns[col_index])
            axes[i, j].set_xlabel("Value")
            axes[i, j].set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def multi_histogram_by_class(df1, df2, n, m, bins=10, title="Histograms", alpha=0.7):
    """
    Plot n by m subplots, each with two histograms from two DataFrames.

    Parameters:
    - df1: First DataFrame
    - df2: Second DataFrame
    - n: Number of rows of subplots
    - m: Number of columns of subplots
    - bins: Number of bins for the histograms (default is 10)
    - alpha: Transparency level of the histograms (default is 0.7)
    """
    
    fig, axes = plt.subplots(n, m, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    for i in range(n):
        for j in range(m):
            column_name = df1.columns[i * m + j]

            axes[i, j].hist(df1[column_name], bins=bins, alpha=alpha, label='DataFrame 1')
            axes[i, j].hist(df2[column_name], bins=bins, alpha=alpha, label='DataFrame 2')

            axes[i, j].set_title(column_name)
            axes[i, j].set_xlabel('Value')
            axes[i, j].set_ylabel('Frequency')

            axes[i, j].legend()

    plt.tight_layout()
    plt.show()