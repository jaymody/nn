import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


class MnistStrokeSequencesClassificationDataset(Dataset):
    def __init__(self, data, ignore_first=True):
        if ignore_first:
            # we may want to ignore the first datapoint in the sequence, since
            # it contains the initial position of the stroke, this gives away
            # spatial information which may be undesirable if we are strictly
            # testing a sequential based model
            self.X = [torch.FloatTensor(d["sequence"][1:]) for d in data]
        else:
            self.X = [torch.FloatTensor(d["sequence"]) for d in data]
        self.y = [torch.tensor(d["digit"]) for d in data]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

    @staticmethod
    def load_train_test_data(filepath):
        with open(filepath) as fi:
            data = json.load(fi)

        train_data = [d for d in data if d["filename"].split("-")[0] == "trainimg"]
        test_data = [d for d in data if d["filename"].split("-")[0] == "testimg"]
        return train_data, test_data

    @staticmethod
    def collate_fn(batch):
        X, y = zip(*batch)
        X = nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
        y = torch.LongTensor(y)
        return X, y

    @staticmethod
    def download_and_process(output_json_path):
        import glob
        import json
        import os
        import tarfile
        import tempfile

        import requests
        from tqdm import tqdm

        def file_to_datapoint(filepath):
            d = {
                "digit": None,
                "sequence": [],
                "num_segments": 0,
                "filename": os.path.basename(filepath),
            }
            with open(filepath) as fi:
                for line in fi:
                    nums = list(map(int, line.split(" ")))

                    d["sequence"].append(nums[10:13])
                    d["num_segments"] += nums[12]

                    digit = nums.index(1)
                    if d["digit"] is None:
                        d["digit"] = digit
                    else:
                        assert digit == d["digit"]
            return d

        print(f"\n\n\n... downloading ...")
        url = "https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/raw/master/sequences.tar.gz"
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise requests.exceptions.HTTPError(f"Failed to download file {url}.")

        with tempfile.TemporaryDirectory() as tmpdirname:
            print(f"\n\n\n... extracting data to {tmpdirname} ...")
            tar = tarfile.open(fileobj=r.raw, mode="r:gz")
            tar.extractall(path=tmpdirname)

            print(f"\n\n\n... processing data ...")
            filepaths = glob.glob(os.path.join(tmpdirname, "sequences", "*targetdata*"))
            data = []
            for filepath in tqdm(filepaths):
                data.append(file_to_datapoint(filepath))

        print(f"\n\n\n... saving to {output_json_path} ...")
        with open(output_json_path, "w") as fo:
            json.dump(data, fo)


class MnistStrokeSequencesImageDataset(MnistStrokeSequencesClassificationDataset):
    def __init__(self, data):
        # we want to ignore the first datapoint in the sequence, since it
        # contains the initial position of the stroke, which we don't want (we
        # only want the dx and dy datapoints when drawing the image)
        sequences = [d["sequence"][1:] for d in data]

        self.X = torch.Tensor(
            np.array(
                [
                    np.array(MnistStrokeSequencesImageDataset.sequence_to_image(seq))
                    for seq in tqdm(sequences)
                ]
            )
        )
        self.y = [torch.tensor(d["digit"]) for d in data]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

    @staticmethod
    def tensor_to_image(tensor):
        from PIL import Image

        image = Image.fromarray(np.array(tensor, dtype=np.uint8))
        return image

    @staticmethod
    def sequence_to_image(sequence):
        from PIL import Image, ImageDraw

        # draw digit
        image = Image.new("L", (64, 64))
        draw = ImageDraw.Draw(image)
        point = (32, 32)
        for x, y, _ in sequence:
            next_point = (point[0] + x, point[1] + y)
            draw.line([point, next_point], fill=255)
            point = next_point

        # crop image to only part containing digit
        image = image.crop(image.getbbox())

        # expand the image to mnist 28 * 28 size
        final_image = Image.new("L", (28, 28))
        final_image.paste(image, ((28 - image.size[0]) // 2, (28 - image.size[1]) // 2))

        return final_image
