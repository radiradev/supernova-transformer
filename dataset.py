import torch
import numpy as np 


class Dataset(torch.utils.data.Dataset):
    #(position_x, position_y, adc_integral_val, adc_peak_val)
    def __init__(
            self, 
            es_path = "data/elastic_scatter.npy", 
            cc_path = "data/charged_current.npy"
        ):
        self.es = self.load_clusters(es_path)
        self.cc = self.load_clusters(cc_path)

        # min len
        min_len = min(len(self.es), len(self.cc))
        self.es = self.es[:min_len]
        self.cc = self.cc[:min_len]
        
        print(f"Number of elastic scatter events: {len(self.es)}")
        print(f"Number of charged current events: {len(self.cc)}")
        self.data = self.mix_and_labels(self.es, self.cc)

    def load_clusters(self, path):
        clusters = np.load(path, allow_pickle=True)
        return clusters

    def __len__(self):
        return len(self.es) + len(self.cc)

    def preprocess_tps(self, data):
        for dimension in range(data.shape[1]):
            # max scale
            max_ = np.max(data[:, dimension])
            data[:, dimension] = data[:, dimension] / max_
        return data


    def mix_and_labels(self, es, cc):
        data = np.vstack((es, cc))

        labels_es = np.zeros(len(es))
        labels_cc = np.ones(len(cc))
        data = self.preprocess_tps(data)
        labels = np.concatenate((labels_es, labels_cc))

        # shuffle data
        idx = np.random.permutation(len(data))
        data = data[idx]
        labels = labels[idx]
        return data, labels
    
    def __getitem__(self, idx):
        data, label = self.data[0][idx], self.data[1][idx]
        return np.array(data, dtype=np.float32), label


if __name__ == "__main__":
    dataset = Dataset()
    print(dataset[0][0])




