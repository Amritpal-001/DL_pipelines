
from amrit.utils.filefolder import get_Flatted_Numpy_Images_from_DataFrame

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import  matplotlib.pyplot as plt

def get_ImageDataset_tSNE(dataframe= None, n_components=180 , perplexity=80.0 , path_column = 'file_path' , label_column = 'target')):
    images, labels = get_Flatted_Numpy_Images_from_DataFrame(dataframe , path_column = path_column , label_column = label_column))
    label_to_id_dict = {v: i for i, v in enumerate(np.unique(labels))}
    id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
    label_ids = np.array([label_to_id_dict[x] for x in labels])
    images_scaled = StandardScaler().fit_transform(images)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(images_scaled)

    tsne = TSNE(n_components=2, perplexity=perplexity)
    tsne_result = tsne.fit_transform(pca_result)
    tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

    visualize_scatter(tsne_result_scaled, label_ids, id_to_label_dict)

def visualize_scatter(data_2d, label_ids, id_to_label_dict):
    plt.grid()

    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color=plt.cm.Set1(label_id / float(nb_classes)),
                    #linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')
    #plt.show()
