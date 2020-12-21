import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC, SVC
from IPython.core.debugger import set_trace


def get_tiny_images(image_paths):
  feats = []

  w = 16; h = 16;
  N = len(image_paths)

  for path in (image_paths):
      image = load_image_gray(path)
      img = cv2.resize(image,(w,h))
      feature = np.reshape(img,(1, w*h))
      feature -= np.mean(feature)
      feature /= np.linalg.norm(feature)
      #print(feature.shape)
      feats.append(feature)

  feats = np.asarray(feats)
  feats = np.reshape(feats,(N, w*h))

  return feats

def build_vocabulary(image_paths, vocab_size):

  dim = 128      # length of the SIFT descriptors that you are going to compute.
  image = load_image_gray(image_paths[0])
  vs = 20
  vb = 9
  _, X = vlfeat.sift.dsift(image, step = vs, size = vb, fast = True)

  for i in range(1,len(image_paths)):
      image = load_image_gray(image_paths[i])
      _, descriptors = vlfeat.sift.dsift(image, step = vs, size = vb, fast = True)
      X = np.vstack((X,descriptors))

  X = np.float32(X)
  vocab = vlfeat.kmeans.kmeans(X, vocab_size)

  return vocab

def get_bags_of_sifts(image_paths, vocab_filename):

  # load vocabulary
  with open(vocab_filename, 'rb') as f:
    vocab = pickle.load(f)

  # dummy features variable
  feats = []

  for path in image_paths:
      image = load_image_gray(path)
      _, descriptors = vlfeat.sift.dsift(image, step = 5, fast = True)
      descriptors = np.float32(descriptors)
      centers = vlfeat.kmeans.kmeans_quantize(descriptors, vocab)
      feature, _ = np.histogram(centers, bins = np.linspace(0, len(vocab), num = len(vocab)+1))
      feature = (feature/np.linalg.norm(feature))**(0.3)
      feats.append(feature)

  feats = np.asarray(feats)
  feats = np.reshape(feats,(len(image_paths), len(vocab)))

  return feats

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
    metric='l2'):

    test_labels = []
    D = sklearn_pairwise.pairwise_distances(test_image_feats, train_image_feats, metric = metric)
    N = D.shape[1]
    k = round(0.1*N)

    for element in D:
        closest_label = np.argmin(element)
        test_labels.append(train_labels[closest_label])

    return  test_labels

def bayes_nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, metric='l2'):
    test_labels = []
    classes = np.unique(train_labels)
    S = np.zeros((test_image_feats.shape[0],len(classes)))

    for i in range(len(classes)):
        k = [ind for ind, value in enumerate(train_labels) if value == classes[i]]
        x = train_image_feats[k]
        for j in range(test_image_feats.shape[0]):
            y = np.reshape(test_image_feats[j], (1, -1))
            D = sklearn_pairwise.pairwise_distances(y, x, metric = metric)
            S[j,i] = np.amin(D)

    for s in S:
        closest_label = np.argmin(s)
        test_labels.append(classes[closest_label])

    return  test_labels

def svm_classify(train_image_feats, train_labels, test_image_feats, kernel):
    # categories
    categories = list(set(train_labels))
    test_labels = []

    # construct 1 vs all SVMs for each category
    #svms = {cat: LinearSVC(random_state=0, tol=1e-5, loss='hinge', C=1) for cat in categories}
    w = []
    h = []
    D = np.zeros((len(categories),test_image_feats.shape[0]))

    for i in range(len(categories)):
        label = categories[i]
        ind = [i for i, entry in enumerate(train_labels) if entry == label]
        y =  (-1)*np.ones(len(train_labels)); y[ind] = 1;

        #svm = LinearSVC(random_state=0, tol=1e-6, loss='hinge', C=1, max_iter = 20000)
        svm = SVC(kernel = kernel, gamma = 0.3, random_state=0, tol=1e-6, C=2, max_iter = 20000)
        svm.fit(train_image_feats,y)
        D[i] = svm.decision_function(test_image_feats)

    D = np.transpose(D)

    for s in D:
        closest_label = np.argmax(s)
        test_labels.append(categories[closest_label])

    return test_labels
