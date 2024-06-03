from __future__ import absolute_import, division, print_function

import os
import argparse
import math
import faiss
import tensorflow as tf
import numpy as np
import pickle

from models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from models.netvlad import netvlad

import config
from prepare_data_custom import get_inference_query_set

from tensorflow.python.client import device_lib


def parse_arguments():

    parser = argparse.ArgumentParser(description='pytorch-NetVlad')
    parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster'])
    parser.add_argument('--batchSize', type=int, default=1, 
            help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
    # parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
    # parser.add_argument('--cacheRefreshRate', type=int, default=1000, 
    #         help='How often to refresh cache, in number of queries. 0 for off')
    # parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
    # parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
    #         help='manual epoch number (useful on restarts)')
    # parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
    # parser.add_argument('--optim', type=str, default='ADAM', help='optimizer to use', choices=['SGD', 'ADAM'])
    # parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
    # parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
    # parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
    # parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
    parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads for each data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
    # parser.add_argument('--dataPath', type=str,
    #                     default='/ssd_data1/lg/pytorch-Netvlad-orig/datasets/240416/train',
    #                     help='Path for centroid data.')
    parser.add_argument('--runsPath', type=str, default='./work_dir/runs/run-99/', help='Path to save runs to.')
    parser.add_argument('--savePath', type=str, default='checkpoints', 
            help='Path to save checkpoints to in logdir. Default=checkpoints/')
    parser.add_argument('--cachePath', type=str, default='./cache/', help='Path to save cache to.')
    parser.add_argument('--resume', type=str, default='./', help='Path to load checkpoint from, for resuming training or testing.')
    parser.add_argument('--ckpt', type=str, default='latest', 
            help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
    parser.add_argument('--evalEvery', type=int, default=1, 
            help='Do a validation set run, and save, every N epochs.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping. 0 is off.')
    # parser.add_argument('--dataset', type=str, default='robotdata',
    #         help='Dataset to use', choices=['pittsburgh','robotdata'])
    parser.add_argument('--arch', type=str, default='resnet18', 
            help='basenetwork to use', choices=['vgg16', 'alexnet', 'resnet18', 'mobilenetv3'])
    parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
    parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
            choices=['netvlad', 'max', 'avg'])
    parser.add_argument('--num_clusters', type=int, default=32, help='Number of NetVlad clusters. Default=64')
    parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
    parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is val', 
            choices=['test', 'test250k', 'train', 'val'])
    parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')
    parser.add_argument('--nNegSample', type=int, default=16)
    parser.add_argument('--nNeg', type=int, default=10)
    parser.add_argument('--super', action='store_true', help='supervised learning mode')
    parser.add_argument('--structFile', type=str, default='./LG_Struct/240513-room2.v2.mat')
    args = parser.parse_args()

    return args


def get_model():
    model = resnet_50()
    if config.model == "resnet18":
        model = resnet_18()
        pool = None
    if config.model == "resnet34":
        model = resnet_34()
        pool = None
    if config.model == "resnet101":
        model = resnet_101()
        pool = None
    if config.model == "resnet152":
        model = resnet_152()
        pool = None
    if config.model == "netvlad":
        model = resnet_18()
        pool = netvlad()

    if pool is not None:
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        pool.build(input_shape=(None, 8, 10, 512)) #!DEBUG

        model.summary()
        pool.summary()
    else:
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        model.summary()

    return model, pool


def triplet_margin_loss(query,postive,negative,margin = 0.1 ** 0.5):
    positive_distance = tf.keras.backend.sum(tf.square(query - postive), axis=-1)
    negative_distance = tf.keras.backend.sum(tf.square(query - negative), axis=-1)
    loss = positive_distance - negative_distance
    loss = tf.maximum(loss + margin, 0.0)
    return loss


if __name__ == '__main__':

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # gpus = device_lib.list_local_devices()
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # parse arguments
    opt = parse_arguments()
    MAT_FILENAME = os.path.basename(opt.structFile).replace('.mat', '')

    with tf.device("/gpu:0"):
        # get the original_dataset
        a=1
        test_data_loader = get_inference_query_set(opt)

        # create model
        encoder_dim = 512 # if resnet18
        model, pool = get_model()

        # load checkpoint
        # ...

    # load db
    cache_path = f"./cache/cache-99/{MAT_FILENAME}.npy"
    pool_size = encoder_dim
    eval_set = test_data_loader
    if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
    if not os.path.isfile(cache_path):
        print('====> Extracting Features')
        dbFeat = np.empty((len(test_data_loader.images), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            image_encoding = model(input)
            vlad_encoding = pool(image_encoding)

            try:
                dbFeat[(iteration - 1), :] = vlad_encoding.numpy()
                if iteration % 50 == 0 or len(test_data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration,
                        len(test_data_loader)), flush=True)
            except:
                a=1

            del input, image_encoding, vlad_encoding
        # del test_data_loader
    else:
        print(f'====> Load Features from cache: {cache_path}')
        dbFeat = np.load(cache_path)

    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')

    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1,5,10]

    _, predictions = faiss_index.search(qFeat, max(n_values))

    #! EVAL DEBUG (240419_test)
    _dict = {
        "eval_set.images": eval_set.images,
        "predictions": predictions,
    }
    os.makedirs(f'{opt.resume}/eval/', exist_ok=True)
    np.save(f'{opt.resume}/eval/{MAT_FILENAME}_dbFeat.npy', dbFeat)
    np.save(f'{opt.resume}/eval/{MAT_FILENAME}_qFeat.npy', qFeat)
    with open(f'{opt.resume}/eval/{MAT_FILENAME}.pickle', mode='wb') as f:
        pickle.dump(_dict, f)

    import sys
    sys.exit()


    # start inference
    # with torch.no_grad():
    startIter = 1
    for epoch in range(config.EPOCHS):
        step = 0
        epoch_loss = 0
        for iteration, (query, positives, negatives, negCounts) in enumerate(infer_data_loader):
            step += 1

            if query is None: continue # in case we get an empty batch

            with tf.GradientTape() as tape:
                B, C, H, W = query.shape
                nNeg = tf.reduce_sum(negCounts, axis=0)
                input = tf.concat([query, positives, negatives], axis=0)

                image_encoding = model(input)
                vlad_encoding = pool(image_encoding)

                vladQ, vladP, vladN = tf.split(vlad_encoding, [B, B, nNeg])

                loss = 0
                for i, negCount in enumerate(negCounts):
                    for n in range(negCount):
                        negIx = int((tf.reduce_sum(negCounts[:i]) + n).numpy())
                        loss += triplet_margin_loss(vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1])

                loss /= tf.cast(nNeg, dtype=tf.float32)

            trainables = model.trainable_weights + pool.trainable_weights
            gradients = tape.gradient(loss, trainables)
            optimizer.apply_gradients(zip(gradients, trainables))

            print(f"[Iter {iteration}] {loss.numpy()[0]:.5f}")
            epoch_loss += loss

        epoch_loss /= len(train_data_loader)
        print(f"Epoch: {epoch+1}/{config.EPOCHS}, \
                loss: {epoch_loss.numpy()[0]:.5f}")

    name = "version1"
    model.save_weights(filepath="saved_model/model", save_format='tf')
    pool.save_weights(filepath="saved_model/pool", save_format='tf')

    model.save('saved_model/model/'+name+'_resnet')
    pool.save('saved_model/pool/'+name+'_pool')

    # Convert to TFLite
    new_model_resnet = tf.keras.models.load_model('saved_model/model/'+name+'_resnet')
    converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/model/'+name+'_resnet')  # path to the SavedModel directory
    tflite_model_resnet = converter.convert()

    new_model_pool = tf.keras.models.load_model('saved_model/pool/' + name + '_pool')
    converter = tf.lite.TFLiteConverter.from_saved_model(
        'saved_model/pool/' + name + '_pool')  # path to the SavedModel directory
    tflite_model_pool = converter.convert()

    # Save the model.
    with open('./saved_model/' + name + '_resnet.tflite', 'wb') as f:
        f.write(tflite_model_resnet)
    # tflite_model_resnet.summary()

    with open('./saved_model/'+name+'_pool.tflite', 'wb') as f:
        f.write(tflite_model_pool)
    # tflite_model_pool.summary()
