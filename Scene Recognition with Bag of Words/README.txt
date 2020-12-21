# CS 6476 project 4: [Scene Recognition with Bag-of-Words](https://www.cc.gatech.edu/~hays/compvision/proj4/)

# Setup
- Install Anaconda / Miniconda
- Create a conda environment using the given file: `conda env create -f environment_<OS>.yml`
- This should create an environment named `cs6476p4`. Activate it using `activate cs6476p4`

#Bells and Whistles
- For vocabulary two sift parameters were measured and used:
step size - referred to as vs; and bin size - referred to as vb.
Both parameters can be changed within build vocabulary function;
Cross validation estimated best ones to be vs = 20 or 30 (for different vocab sizes) and vb = 9 (6 performs better if vocab size smaller than 20 is played with)

- For bag of sifts best step size s, which can also be changed within the function, was estimated to be equal to 5 for best performance, and is set that way for now

- For SVM kernel linear and rbf were tested, this parameter therefore along with C and gamma could be tempered with.
For best performance use rbf, C = 2 and gamma = 0.3 with default vocab, for vocabs bigger than 400-500, use gamma = 1

- NBNN was also implemented and could be called by adding 'bayes_' prefix to nearest neighbor function, yet since I didn't introduce gist or any other additional descriptors it is the same as calling NN

#Experimental design
- proj4_cross.ipynb - dummy used for validation and cross validation (utils.py) was changed accordingly
