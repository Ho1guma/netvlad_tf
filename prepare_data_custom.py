import tensorflow as tf
import config
import pathlib
from config import image_height, image_width, channels

import os
import glob
import copy
from collections import namedtuple
from PIL import Image
import numpy as np
import math
from scipy.io import loadmat


dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
                                   'dbImage', 'qImage', 'augImage',
                                   'numDb', 'numQ'])
TestdbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
    'dbImage', 'qImage', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr','gt'])

def input_transform_tf(image):

    image = tf.convert_to_tensor(np.array(image), dtype=tf.float32)

    # Resize image
    image = tf.image.resize(image, size=(320, 240))

    # Normalize image
    image = tf.image.per_image_standardization(image)

    return image

def parse_dbStruct(opt):
    a=1
    augPath = '/'.join(opt.dataPath.split('/')[:-1] + ["aug_train_240501"]) # aug_train_colorjitter
    if os.path.isdir(augPath):
        aug_file_dict = {}
        for folder in os.listdir(augPath):
            aug_file_dict[folder] = glob.glob(os.path.join(augPath, folder) + "/*")
        augImage = aug_file_dict
    else:
        augImage = None

    folder_list = os.listdir(opt.dataPath)
    file_list = []
    for folder in folder_list:
        file_list.append(glob.glob(os.path.join(opt.dataPath,folder) +"/*"))
        # file_list.append(tuple(join(opt.dataPath,folder)))
    dataset = opt.dataset

    # train, val, test
    whichSet = opt.mode

    # dbImage: 전체 이미지 경로
    dbImage = file_list

    qImage = sum(file_list,[])          # [folder_num]
    # utmQ = matStruct[4].T

    qImage = [x for x in qImage if "jpg" in x] #!DEBUG

    numDb = len(dbImage)            #
    numQ = len(qImage)              #

    return dbStruct(whichSet, dataset, dbImage, qImage, augImage, numDb, numQ)


def load_and_preprocess_image(img_path):
    # read pictures
    img_raw = tf.io.read_file(img_path)
    # decode pictures
    img_tensor = tf.image.decode_jpeg(img_raw, channels=channels)
    # resize
    img_tensor = tf.image.resize(img_tensor, [image_height, image_width])
    img_tensor = tf.cast(img_tensor, tf.float32)
    # normalization
    img = img_tensor / 255.0
    return img

def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    return all_image_path, all_image_label


def get_training_query_set(opt):
    # structFile = join(opt.struct_dir, 'pitts30k_train.mat')
    a=1
    input_transform = input_transform_tf
    return QueryDatasetFromStruct(input_transform=input_transform, opt=opt)

def get_inference_query_set(opt):
    a=1
    input_transform = input_transform_tf
    return TestDatasetFromStruct(input_transform=input_transform, opt=opt)


def get_dataset(dataset_root_dir):
    a=1

    all_image_path, all_image_label = get_images_and_labels(data_root_dir=dataset_root_dir)
    # print("image_path: {}".format(all_image_path[:]))
    # print("image_label: {}".format(all_image_label[:]))
    # load the dataset and preprocess images
    image_dataset = tf.data.Dataset.from_tensor_slices(all_image_path).map(load_and_preprocess_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    image_count = len(all_image_path)

    return dataset, image_count


def generate_datasets():
    train_dataset, train_count = get_dataset(dataset_root_dir=config.train_dir)

    # read the original_dataset in the form of batch
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)

    return train_dataset, train_count


# ===== Datasets =====
class QueryDatasetFromStruct(tf.keras.utils.Sequence):
    def __init__(self, opt, input_transform=None, batch_size=4):
        super().__init__()
        self.input_transform = input_transform
        # self.margin = opt.margin

        self.dbStruct = parse_dbStruct(opt)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = opt.nNegSample  # number of negatives to randomly sample
        self.nNeg = opt.nNeg  # number of negatives used for training
        self.super = opt.super
        self.key = []

        self.batch_size = batch_size

        # queries의 positive index
        self.nontrivial_positives = []
        self.potential_negatives = []

        self.queries = list(range(0,self.dbStruct.numDb))
        self.key = list(range(0,self.dbStruct.numDb))

        # qImage : [folder_num]
        # nontrivial_positives : [folder_num,]
        tmp = 0
        for i, posi in enumerate(self.dbStruct.dbImage):
            self.nontrivial_positives.append(list(range(tmp, tmp+len(posi)))) #[i]*len(posi)
            # query 의 negative index
            # self.potential_negatives[i] = list(range(0,self.dbStruct.numQ)).remove(i)
            self.key[tmp:tmp+len(posi)] = [i] * len(posi)
            tmp += len(posi)

        if getattr(self.dbStruct, "augImage", None) is not None:
            self.augImage = self.dbStruct.augImage

        self.negCache =[np.empty((0,)) for _ in range(self.dbStruct.numQ)]

        self.cache = None  # filepath of HDF5 containing feature vectors for images
        self.indices = [x for x in range(0, len(self.queries))]

    def __len__(self):
        return math.ceil(len(self.queries) / self.batch_size)

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        data = [self.get_item(x) for x in indices]

        query = tf.stack([x[0] for x in data])
        positives = tf.stack([x[1] for x in data])
        negatives = tf.concat([x[2] for x in data], axis=0)
        negCounts = [len(x[2]) for x in data]

        return query, positives, negatives, negCounts

    def get_item(self, index):

        # 1) randomly select single positive index from same group
        posIndice = copy.deepcopy(self.nontrivial_positives[self.key[index]])
        posIndice.remove(index)
        posIndex = np.random.choice(posIndice,1) # random하게 1개 선택

        # 2) randomly select nNegSample negative indices from different groups
        allIndex = copy.deepcopy(sum(self.nontrivial_positives,[]))
        [allIndex.remove(i) for i in posIndice]
        allIndex.remove(index)
        negIndices = np.random.choice(allIndex, self.nNegSample) # random으로 nNegSample개 선택

        # 3) randomly select single positive index from augmented image list
        _query = os.path.basename(self.dbStruct.qImage[index]).split('.')[0]
        posFilepaths = self.augImage[_query]
        # posFilepaths = [x for x in sum(self.augImage, []) if _query in x]
        assert len(posFilepaths) == 8, f"Wrong number of augmented images: {_query}"
        posFilepath = np.random.choice(posFilepaths, 1)[0]
        a=1

        # 4) load images and serve
        query = Image.open(self.dbStruct.qImage[index])
        a=1
        if getattr(self.dbStruct, "augImage", None) is not None:
            positive = Image.open(posFilepath)
        else:
            positive = Image.open(self.dbStruct.qImage[posIndex[0]])

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            negative = Image.open(self.dbStruct.qImage[negIndex])
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)

        # negatives = torch.stack(negatives, 0)
        negatives = tf.stack(negatives)

        if not isinstance(negIndices, list):
            negIndices = negIndices.tolist()

        return query, positive, negatives, [index, posIndex] + negIndices #.tolist()


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        # Load and preprocess data here
        # Example: read images and labels from disk, apply augmentation, etc.
        X = np.zeros((len(list_IDs_temp), 32, 32, 3))
        y = np.zeros((len(list_IDs_temp), 1))

        # Placeholder for actual data loading and preprocessing
        for i, ID in enumerate(list_IDs_temp):
            # Load data and preprocess
            pass

        return X, y


class TestDatasetFromStruct(tf.keras.utils.Sequence):
    def __init__(self, opt, input_transform=None, batch_size=1):
        super().__init__()
        assert batch_size == 1
        self.batch_size = batch_size

        self.input_transform = input_transform
        self.dbStruct = self.parse_TestdbStruct(opt.structFile)

        self.images = [x.replace(" ", "") for x in self.dbStruct.dbImage]
        self.images += [x.replace(" ", "") for x in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

        self.indices = [x for x in range(0, len(self.images))]

        a=1
    
    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        data = []
        for i in indices:
            img = Image.open(self.images[i])
            if self.input_transform:
                try:
                    img = self.input_transform(img)
                except:
                    a=1
            data.append(img)
        data = [data] if len(data) == 1 else data

        images = tf.stack([x[0] for x in data])

        return images, index

    def parse_TestdbStruct(self, path):
        matStruct = loadmat(path)

        dataset = 'robotdata'
        data_dir = "/ssd_data1/lg/pytorch-Netvlad-orig/"

        whichSet = matStruct['whichSet']

        dbImage = [os.path.join(data_dir, f[0].item()) for f in matStruct['dbImage']]
        #utmDb = matStruct[2].T

        qImage = [os.path.join(data_dir, f[0].item()) for f in matStruct['qImage']]
        #utmQ = matStruct[4].T

        numDb = matStruct['numDb'].item()
        numQ = matStruct['numQ'].item()

        posDistThr = matStruct['posDistThr'].item()
        posDistSqThr = matStruct['posDistSqThr'].item()
        nonTrivPosDistSqThr = matStruct['nonTrivPosDistSqThr'].item()
        gt=  matStruct['gt']

        return TestdbStruct(whichSet, dataset, dbImage, qImage,
                numDb, numQ, posDistThr,
                posDistSqThr, nonTrivPosDistSqThr,gt)
