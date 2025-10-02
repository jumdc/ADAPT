"""Create the extraction of features."""

import os
import subprocess
from math import ceil
from copy import deepcopy

import hydra
import torch
import pandas as pd
import numpy as np
from PIL import Image
import pyrootutils

from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

from tqdm import tqdm

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.models.modules.make_vision import make_hiera
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class VideoDataset(Dataset):
    """creates a dataset for video classification.

    fps of 5 frames per second is used.
    1280 x 720 resolution is used.

    Parameters
    ----------
    cfg : DictConfig
        configuration file.
    video_id : str
        video id.
    """

    def __init__(self, cfg, videos_ids) -> None:
        """initialize the dataset."""
        super().__init__()
        self.cfg = cfg
        self.img_dim = (224, 224)
        self.video_ids = videos_ids
        self.all_files = pd.read_csv(
            os.path.join(cfg.paths.data, cfg.dataset.path_labels),
            sep=",",
            header=0,
            index_col=0,
        ).dropna()
        self.num_classes = 2
        idx_label = "binary-stress"
        self.labels = deepcopy(self.all_files)
        self.labels = self.labels[idx_label]
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.img_dim, antialias=True),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )
        paths = self._get_videos_path(
            cfg=cfg, ids=videos_ids, labels=self.labels.index.values
        )
        (self.data, self.positive_idx, self.negative_idx, self.incoming_idx) = (
            self._get_samples_from_videos(paths)
        )

        self.number_positives = len(self.positive_idx)
        self.number_negatives = len(self.negative_idx)
        self.number_incoming = len(self.incoming_idx)

    @staticmethod
    def _get_videos_path(cfg, ids, labels):
        """get videos path."""
        data = []
        path_frames = os.path.join(cfg.paths.data, cfg.dataset.path_frames)
        for directory in os.listdir(path_frames):
            id_file = directory.split(".")[0]
            ## attention make sure we want to do the split based on files and not ids.
            if id_file in ids and id_file in labels:
                data.append(os.path.join(path_frames, directory))
        data = list(set(data))
        return data

    def _get_samples_from_videos(
        self,
        video_paths: list,
    ):
        """get samples from videos."""
        data, values, labels, ids = [], [], [], []
        if not os.path.exists(
            os.path.join(self.cfg.paths.data, self.cfg.dataset.path_frames)
        ):
            os.makedirs(os.path.join(self.cfg.paths.data, self.cfg.dataset.path_frames))
        frames_dir = os.listdir(
            os.path.join(self.cfg.paths.data, self.cfg.dataset.path_frames)
        )
        for video_path in video_paths:
            video = video_path.split("/")[-1].split(".")[0]
            if video not in frames_dir:
                self._create_frames_from_video(
                    video_path,
                    video,
                    os.path.join(self.cfg.paths.data, self.cfg.dataset.path_frames),
                )
            tamp_path = os.listdir(
                os.path.join(self.cfg.paths.data, self.cfg.dataset.path_frames, video)
            )
            values_ = []
            for file in tamp_path:
                if file[-3:] == "jpg":  # here jpg instead of png
                    values_.append(
                        os.path.join(
                            self.cfg.paths.data,
                            self.cfg.dataset.path_frames,
                            video,
                            file,
                        )
                    )
            ## find label and ids
            values.append(values_)
            ids.append(video)
            labels.append(self.labels.loc[video])
        window = self.cfg.dataset.video.window
        ids_sequences, labels_sequences, paths_sequences = [], [], []
        for v_idx in range(len(values)):
            nb_idx = 0
            if window is not None:
                for idx in range(
                    0, len(values[v_idx]) + 1, self.cfg.dataset.video.stride
                ):
                    if len(values[v_idx][idx - window : idx]) == window:
                        id_final_video = ids[v_idx].replace("_", ":")
                        ids_sequences.append(
                            f"{id_final_video}:{nb_idx}:{labels[v_idx]}"
                        )
                        labels_sequences.append(labels[v_idx])
                        paths_sequences.append(values[v_idx][idx - window : idx])
                        nb_idx += 1
            else:
                id_final_video = ids[v_idx].replace("_", ":")
                ids_sequences.append(f"{id_final_video}:{nb_idx}:{labels[v_idx]}")
                labels_sequences.append(labels[v_idx])
                paths_sequences.append(values[v_idx])
        ids_sequences = np.asarray(ids_sequences)
        data = (ids_sequences, labels_sequences, paths_sequences)
        positives_sequences_idx = [i for i in range(len(data[1])) if data[1][i] == 1]
        negatives_sequences_idx = [i for i in range(len(data[1])) if data[1][i] == 0]
        incomplete_sequences_idx = [i for i in range(len(data[1])) if data[1][i] == 2]
        return (
            data,
            positives_sequences_idx,
            negatives_sequences_idx,
            incomplete_sequences_idx,
        )

    @staticmethod
    def _create_frames_from_video(video_path: str, video: str, path_frames: str):
        """create frames from video."""
        os.makedirs(os.path.join(path_frames, video), exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-vf",
                "fps=5",
                os.path.join(path_frames, video, f"{video}_frame-%03d.jpg"),
            ]
        )

    def _get_sequence(self, path):
        """get the sequence from the video"""
        frames = torch.stack([ToTensor()(Image.open(frame)) for frame in path])
        frames = self.transform(frames)
        frames = frames.permute(1, 0, 2, 3)
        return frames

    def __len__(self):
        """size of the dataset"""
        return len(self.data[0])

    def __getitem__(self, idx):
        """get item."""
        ids, label, path = self.data[0][idx], self.data[1][idx], self.data[2][idx]
        # the step may depend on the lenght of the sequence.
        if self.cfg.dataset.video.step is not None:
            ## step must be given something.
            path = path[:: self.cfg.dataset.video.step]
        else:
            step = ceil(len(path) / 16)
            path = path[::step]
        label = int(label)
        value = [0.0] * self.num_classes
        value[label] = 1
        y = torch.Tensor(value)
        output = {}
        output["id"] = ids
        output["val"] = self._get_sequence(path)
        output["label"] = y
        return output


@hydra.main(
    version_base="1.2", config_path="configs", config_name="video-extract.yaml"
)
def main(cfg) -> bool:
    """Compute features for a given model."""
    launches = os.listdir(os.path.join(cfg.paths.data, cfg.dataset.path_frames))
    launches.sort()
    feature_extractor, _ = make_hiera(True, directory=None)
    step = 10
    i = 0
    ids, targets, features = [], [], []
    for i in tqdm(range(0, len(launches), step)):
        # for i in tqdm(range(0, 5, step)):
        videos = VideoDataset(cfg=cfg, videos_ids=launches[i : i + step])
        datamodule = torch.utils.data.DataLoader(
            videos, batch_size=cfg.dataset.hyperparams.batch_size, num_workers=5
        )
        id, target, feature = compute_features(
            feature_extractor=feature_extractor, datamodule=datamodule
        )
        ids.extend(id)
        targets.extend(target)
        features.extend(feature)
    data = list(zip(ids, targets, features))
    ok = save_txt(data=data, filename=os.path.join(cfg.paths.data, cfg.name))
    return ok


def compute_features(feature_extractor, datamodule) -> tuple:
    """
    Compute the features and store them in a file.

    Parameters
    ----------
    feature_extractor : VideoClassification
        Video model.
    datamodule : VideoDatamodule
        Video datamodule.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, ids, targets = [], [], []
    feature_extractor.to(device)
    for data in datamodule:
        ids.extend(data["id"])
        features.append(feature_extractor(data["val"].to(device)).cpu())
        targets.append(data["label"])
    features = torch.cat(features).numpy().tolist()
    targets = torch.argmax(torch.cat(targets), dim=1).numpy().squeeze().tolist()
    return (ids, targets, features)


def save_txt(data: list, filename: str) -> bool:
    """save txt."""
    with open(filename, "w") as f:
        f.writelines(
            (
                ":".join(str(item) for item in items) + "\n"
                if i < len(data) - 1
                else ":".join(str(item) for item in items)
            )
            for i, items in enumerate(data)
        )
    return True


if __name__ == "__main__":
    main()
