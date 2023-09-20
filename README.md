# German Character Recognition Dataset

This code trains a CNN with PyTorch to build a classifier for the German Character Recognition Dataset.
The dataset contains 282,472 grayscale images, each measuring 40 x 40 pixels, depicting a diverse range
of 82 distinct German characters and mathematical symbols. The dataset can be found
and downloaded [here](https://www.kaggle.com/datasets/thomassedlmeyr/german-character-recognition-dataset).
On digit recognition the classifier achieves roughly 99% ACC, AUC and MCC (Matthew's Correlation Coefficient)
on the test set. 

## Dependencies
You only need to set up the conda environment from the environment.yml file.

```
conda env create -f environment.yml
```
## How to use the code
There are two options how you can start the training. You can either run the code in a Jupyter Notebook training.ipynb
or you can run the python script training.py. Both files are located in the folder code. 


## Authors

* Thomas Sedlmeyr [GitHub](https://github.com/ThomasSedlmeyr), [LinkedIn](https://www.linkedin.com/in/thomas-sedlmeyr/)
* Philip Haitzer [GitHub](https://github.com/PhilipHaitzer)

## License

This project is licensed under the [Attribution-NonCommercial 4.0 International] License - see the LICENSE.md file for details

