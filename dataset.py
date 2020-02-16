import torch.utils.data as data
import numpy as np
import pickle as pkl
import os
import os.path
import gzip
import errno
import tensorflow as tf
import torch
import os
import glob
import sys
from utils import AttrDict
#from specs import machine_dependent_specs as m_specs
import numpy as np
import re


from torchvision.datasets.folder import is_image_file, default_loader, find_classes, \
    IMG_EXTENSIONS
from torchvision.datasets.utils import download_url


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


class MergedDataset(data.Dataset):
    """
    Merged multiple datasets into one. Sample together.
    """

    def __init__(self, *datasets):
        self.datasets = datasets
        assert all(len(d) == self.__len__() for d in self.datasets)

    def __getitem__(self, index):
        return [d[index] for d in self.datasets]

    def __len__(self):
        return len(self.datasets[0])

    def __repr__(self):
        fmt_str = ''
        for dataset in self.datasets:
            fmt_str += dataset.__repr__() + '\n'
        return fmt_str


def make_dataset(dir, class_to_idx):
    actions = []
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            # if root[-2:] not in ['66', '67', '68']:
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                if fname == 'actions.npy':
                    path = os.path.join(root, fname)
                    actions.append(np.load(path))
                    actions[-1][-1, 4] = 0.0

    return images, np.concatenate(actions, axis=0)


def make_pair(imgs, resets, k, get_img, root):
    """
    Return a list of image pairs. The pair is picked if they are k steps apart,
    and there is no reset from the first to the k-1 frames.
    Cases:
        If k = -1, we just randomly pick two images.
        If k >= 0, we try to load img pairs that are k frames apart.
    """
    if k < 0:
        return list(zip(imgs, np.random.permutation(imgs)))

    filename = 'imgs_skipped_%d.pkl' % k
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f)

    image_pairs = []
    for i, img in enumerate(imgs):
        if np.sum(resets[i:i + k]) == 0 and (get_img(imgs[i + k][0]) - get_img(img[0])).abs().max() > 0.5:
            image_pairs.append((img, imgs[i + k]))
    with open(filename, 'wb') as f:
        pkl.dump(image_pairs, f)
    return image_pairs


class ImagePairs(data.Dataset):
    """
    A copy of ImageFolder from torchvision. Output image pairs that are k steps apart.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        n_frames_apart (int): The number of frames between the image pairs. Fixed for now.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        img_pairs (list): List of pairs of (image path, class_index) tuples
    """

    url = 'https://drive.google.com/uc?export=download&confirm=ypZ7&id=10xovkLQ09BDvhtpD_nqXWFX-rlNzMVl9'

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, n_frames_apart=1, download=False):
        self.root = root
        if download:
            self.download()

        classes, class_to_idx = find_classes(root)
        imgs, actions = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        resets = 1. - actions[:, -1]
        assert len(imgs) == len(resets)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        img_pairs = make_pair(imgs, resets, n_frames_apart, self._get_image, self.root)
        self.img_pairs = img_pairs

    def _get_image(self, path):
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _check_exists(self):
        return os.path.exists(self.processed_folder)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    def download(self):
        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # for url in self.urls:
        # filename = self.url.rpartition('/')[2]
        filename = "rope"
        file_path = os.path.join(self.raw_folder, filename)
        # import ipdb;ipdb.set_trace()
        download_url(self.url, root=self.raw_folder, filename=filename, md5=None)
        self.extract_gzip(gzip_path=file_path, remove_finished=False)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        output = []
        for path, target in self.img_pairs[index]:
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            output.append((img, target))
        return output

    def __len__(self):
        return len(self.img_pairs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class KeyInSets():
    
    def __init__(self):
        #super().__init__()
        #tf.enable_eager_execution()
        data_dir = '/NAS/scratch/robotic_video/KeyIn/TOP_gtimestep' + "/train"
        #import pdb; pdb.set_trace()
        tfrecord_files = sorted(glob.glob(data_dir + '/*.tfrecord'))
        self.raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
        self.dataset_config = AttrDict(input_width=64, input_height=64,
                            channels=3, num_actions=4, max_seq_length=40, im_width=64, im_heigh=64, num_abs_actions=1)
        self.data_iter = iter(self.raw_dataset)
        
    def get_batch_data(self, batch_size=64):
        #import pdb; pdb.set_trace()
        try:
            while True:
                batch_o = []
                batch_o_next = []
                data_iter = self.get_data()
                for i in range(batch_size):
                    o, o_next = next(data_iter)
                    batch_o.append(o)
                    batch_o_next.append(o_next)
                tensor_o = torch.cat(batch_o).view(batch_size, -1, batch_o[0].shape[1], batch_o[0].shape[2])
                tensor_o_next = torch.cat(batch_o).view(batch_size, -1, batch_o[0].shape[1], batch_o[0].shape[2])
                yield [(tensor_o, None), (tensor_o_next, None)]
        except StopIteration: # this happens because next(data_iter) is exhausted
            self.data_iter = iter(self.raw_dataset)

    def get_plan_data(self):
        try:
            while True:
                for rec in self.data_iter:
                    results = self.parse_top(rec, self.dataset_config)
                    imgs = torch.from_numpy(results[0].numpy().transpose(0, 3, 1, 2)) / 255 # float type, put range to (0, 1)
                    goal_timestep = results[1].numpy()
                    yield imgs, goal_timestep
        except StopIteration:
            self.data_iter = iter(self.raw_dataset)


    def get_data(self, seed=None):
        # try:
        #     while True:
        for rec in self.data_iter:
            # rec = next(self.data_iter)
            # example = tf.train.Example()
            # example.ParseFromString(rec.numpy())
            # keys = sorted(example.features.feature.keys())
            # imgs, max_valid_idx
            results = self.parse_top(rec, self.dataset_config)
            # randomly take out consecutive images as data 
            take_idx = np.random.randint(low=0, high=min(results[1].numpy(),results[2])-1)
            imgs = torch.from_numpy(results[0].numpy().transpose(0, 3, 1, 2)) / 255 # float type, put range to (0, 1)
            yield imgs[take_idx], imgs[take_idx+1]
        # except StopIteration:
        #     self.data_iter = iter(self.raw_dataset)




    def _parse_bairlike_generic(self, example_proto, dataset_config,
                              feature_templates_and_shapes):
        """Parses the BAIR dataset, fuses individual frames to tensors."""
        features = {}  # fill all features in feature dict
        for key, feat_params in feature_templates_and_shapes.items():
            for frame in range(dataset_config.max_seq_length):
                if feat_params["type"] == tf.string:
                    feat = tf.VarLenFeature(dtype=tf.string)
                else:
                    feat = tf.FixedLenFeature(dtype=feat_params["type"],
                                            shape=feat_params["shape"])
                features.update({feat_params["name"].format(frame): feat})
        parsed_features = tf.parse_single_example(example_proto, features)

        # decode frames and stack in video

        def process_feature(feat_params, frame):
            feat_tensor = parsed_features[feat_params["name"].format(frame)]
            if feat_params["type"] == tf.string:
                feat_tensor = tf.sparse_tensor_to_dense(feat_tensor, default_value="")
                # feat_tensor = tf.decode_raw(feat_tensor, tf.float32)
                feat_tensor = tf.cast(tf.decode_raw(feat_tensor, tf.uint8), tf.float32)
                feat_tensor = tf.reshape(feat_tensor, feat_params["shape"])
            return feat_tensor

        parsed_seqs = {}
        for key, feat_params in feature_templates_and_shapes.items():
            if "dim" in feat_params and feat_params["dim"] == 0:
                feat_tensor = process_feature(feat_params, 0)
                parsed_seqs.update({key: feat_tensor})
            else:
                frames = []
                for frame in range(dataset_config.max_seq_length):
                    feat_tensor = process_feature(feat_params, frame)
                    frames.append(feat_tensor)
                parsed_seqs.update({key: tf.stack(frames)})

        return parsed_seqs

    def parse_top(self, example_proto, dataset_config):
        # specify feature templates that frame numbers will be filled into
        img_shape = (dataset_config.input_height, dataset_config.input_width, dataset_config.channels,)
        feature_templates_and_shapes = {
            "goal_timestep": {"name": "goal_timestep", "shape": (1,), "type": tf.int64, "dim": 0},
            "images": {"name": "{}/image_view0/encoded", "shape": img_shape, "type": tf.string, "dim": 1},
            "actions": {"name": "{}/action", "shape": (dataset_config.num_actions,), "type": tf.float32, "dim": 1},
            "abs_actions": {"name": "{}/is_key_frame", "shape": (1,), "type": tf.int64, "dim": 1},
            "is_key_frame": {"name": "{}/is_key_frame", "shape": (1,), "type": tf.int64, "dim": 1}
        }

        parsed_seqs = self._parse_bairlike_generic(example_proto, dataset_config,
                                                                feature_templates_and_shapes)
        data_tensors = AttrDict(goal_timestep=parsed_seqs["goal_timestep"])
        parsed_seqs["actions"] = parsed_seqs["actions"] / 2  # rescale actions
        return parsed_seqs['images'], parsed_seqs["goal_timestep"],dataset_config.max_seq_length
        #return parsed_seqs["images"], parsed_seqs["actions"], \
                #parsed_seqs["abs_actions"], data_tensors, dataset_config.max_seq_length



if __name__ == "__main__":
    d = KeyInSets()

