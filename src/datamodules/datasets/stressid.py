"""StressID dataset"""

import os
import logging
from copy import deepcopy
from itertools import repeat
from multiprocessing import Pool

import torch
import librosa
import numpy as np
import pandas as pd
import torch.utils.data


from src.utils.preprocessing.audio_preprocessing import audio_preprocess
from src.utils.training.augmentations import MultiModalAugmentations


class StressID(torch.utils.data.Dataset):
    """the StressID dataset"""

    def __init__(self, cfg, ids, stage="anchor") -> None:
        """Initialize the StressID dataset

        Parameters
        ----------
        cfg : dict
            Configuration dictionary
        stage : str, optional
            Stage of the dataset, by default "anchor"
        split : str, optional
            Split of the dataset, by default "train"
        """
        super().__init__()
        self.cfg = cfg
        self.ids = ids
        self.stage = stage
        self.mask = None
        self.physio, self.video, self.audio = None, None, None
        labels = pd.read_csv(
            os.path.join(cfg["paths"]["data"], cfg["multimodal"]["path_labels"]),
            sep=",",
            header=0,
            index_col=0,
        ).dropna()["binary-stress"]
        modality_ids, self.mapping = [], []
        if self.stage == "contrastive":
            self.augmentations = MultiModalAugmentations(
                modalities=self.cfg["multimodal"]["modalities"],
                modality_dropout=self.cfg["model"]["contrastive_loss"][
                    "modality_dropout"
                ],
                noise_sigma=self.cfg["model"]["contrastive_loss"]["noise_sigma"],
            )
        self.physio_map = deepcopy(cfg["multimodal"]["modalities"])
        self.get_physio_audio(labels, cfg)
        if (
            len(
                list(
                    set(cfg["multimodal"]["modalities"])
                    & set(["ECG", "EDA", "RR", "AUDIO"])
                )
            )
            > 0
        ):
            if (
                len(
                    list(
                        set(cfg["multimodal"]["modalities"]) & set(["ECG", "EDA", "RR"])
                    )
                )
                > 0
            ):
                self.mapping.append("PHYSIO")
                modality_ids.append([self.physio[0].tolist(), self.physio[1]])
            if "AUDIO" in cfg["multimodal"]["modalities"]:
                modality_ids.append([self.audio[0].tolist(), self.audio[1]])
                self.mapping.append("AUDIO")
                self.physio_map.remove("AUDIO")
        if "VIDEO" in cfg["multimodal"]["modalities"]:
            if self.cfg["multimodal"]["video"]["precomputed_video"]:
                self.video = StressID.get_videos(
                    self.ids,
                    labels,
                    os.path.join(
                        cfg["paths"]["data"], cfg["multimodal"]["path_features_video"]
                    ),
                )
                self.mapping.append("VIDEO")
                modality_ids.append([self.video[0].tolist(), self.video[1]])
                self.physio_map.remove("VIDEO")

        self.mean_physio = []
        self.std_physio = []
        for modality in self.physio_map:
            self.mean_physio.append(self.cfg.multimodal[modality]["global-mean"])
            self.std_physio.append(self.cfg.multimodal[modality]["global-std"])

        ### merge ###
        keep_missing = (
            False
            if (stage in ["anchor", "contrastive"])
            else self.cfg["multimodal"]["keep_missing"]
        )
        if len(self.mapping) > 1:
            self.indices, self.labels = self.merge(
                modality_ids, keep_missing=keep_missing
            )
        else:
            self.indices = modality_ids[0][0]
            self.labels = labels

    @staticmethod
    def merge(lists: list, keep_missing=False):
        """
        Merge the different modalities

        Parameters
        ----------
        lists : list
            List of modalities ids
        keep_missing : bool, optional
            Keep missing values, by default False

        Returns
        -------
        tuple : (np.ndarray, np.array)
            Indices, labels
        """
        sets = [set(lst[0]) for lst in lists]
        all_values = set.union(*sets)
        result, labels = [], []
        for val in all_values:
            all_in = all([val in lst[0] for lst in lists])
            if all_in or keep_missing:
                indices = tuple(
                    sublist[0].index(val) if val in sublist[0] else None
                    for sublist in lists
                )
                labels.append(
                    [
                        lists[idx_list][1][indice]
                        for idx_list, indice in enumerate(indices)
                        if indice is not None
                    ][0]
                )
                result.append(indices)
        return result, labels

    @staticmethod
    def select_sequences(X, y, files: np.ndarray, files_to_keep: list):
        """
        Select the sequences to keep

        Parameters
        ----------
        X : np.ndarray or list
            Data
        y : np.array
            Labels
        files : np.ndarray
            Files
        files_to_keep : list
            Files to keep

        Returns
        -------
        tuple : (np.ndarray, np.array, iterable)
            Files, labels, data
        """
        ids = np.array(["_".join(id.split(":")[0:2]) for id in files])
        selected = np.where(np.isin(ids, files_to_keep))[0]
        X = X[selected] if isinstance(X, np.ndarray) else [X[i] for i in selected]
        y = y[selected]
        files = files[selected]
        return (files, y, X)

    def get_physio_audio(self, labels, cfg):
        """Get the physiological data

        Parameters
        ----------
        labels : pd.DataFrame
            Labels of the data
        cfg : dict
            Configuration dictionary
        """
        path_physio = os.path.join(
            cfg["paths"]["data"], cfg["multimodal"]["physiological_root"]
        )
        path_audio = os.path.join(cfg["paths"]["data"], cfg["multimodal"]["audio_root"])
        files = [
            f
            for f_sub in [f_names for _, _, f_names in os.walk(path_physio)]
            for f in f_sub
        ]
        files_audio = [
            f
            for f_sub in [f_names for _, _, f_names in os.walk(path_audio)]
            for f in f_sub
        ]
        tamp_X, tamp_ids, tamp_ys, tamp_audio, tamp_ids_audio, tamps_ys_audio = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        ### extract physio & audio from files ###
        for file in files:
            if not file.startswith("."):
                id_file = file.split(".")[0]
                tamp = []
                if id_file in labels.index.values.tolist():
                    tamp_ids.append(id_file.replace("_", ":"))
                    tamp_ys.append(labels.loc[id_file])
                    path = os.path.join(
                        path_physio, file.split("_")[0], file  # num of the folder
                    )
                    file = pd.read_csv(path, sep=",")
                    if "ECG" in cfg["multimodal"]["modalities"]:
                        tamp.append(file["ECG"].values.tolist())
                    if "EDA" in cfg["multimodal"]["modalities"]:
                        tamp.append(file["EDA"].values.tolist())
                    if "RR" in cfg["multimodal"]["modalities"]:
                        tamp.append(file["RR"].values.tolist())
                    if len(tamp) > 0:
                        tamp_X.append(tamp)
        #### AUDIO ####
        if "AUDIO" in cfg["multimodal"]["modalities"]:
            ### to complete ###
            for f in files_audio:
                if not f.startswith("."):
                    id_file = f.split(".")[0]
                    if id_file in labels.index.values.tolist():
                        tamp_ids_audio.append(id_file.replace("_", ":"))
                        tamps_ys_audio.append(labels.loc[id_file])
                        ## did not read the file ??
                        folder = f.split("_")[0]
                        sound, _ = librosa.load(
                            os.path.join(path_audio, folder, f), sr=None
                        )
                        tamp_audio.append(sound)

        ### sliding window - if necessary ###
        if cfg["multimodal"]["hyperparams"]["sliding_window"]:
            if "AUDIO" in cfg["multimodal"]["modalities"]:
                args = zip(
                    tamp_audio,
                    tamps_ys_audio,
                    tamp_ids_audio,
                    repeat(cfg["multimodal"]["audio_hyperparams"]["window_audio"]),
                    repeat(cfg["multimodal"]["audio_hyperparams"]["stride_audio"]),
                )
                with Pool(
                    processes=cfg["multimodal"]["hyperparams"]["num_workers"]
                ) as pooling:
                    res = pooling.starmap(StressID.sliding, args)
                    X_audio = np.concatenate(
                        [mts for mts, _, _ in res if mts is not None], axis=0
                    )
                    y = np.concatenate([y for _, y, _ in res if len(y) > 0], axis=0)
                    ids = np.concatenate(
                        [files for _, _, files in res if len(files) > 0], axis=0
                    )
                    ids, y, X_audio = StressID.select_sequences(
                        X_audio, y, ids, self.ids
                    )
                    self.audio = (ids, y, X_audio)
            if (
                len(
                    list(
                        set(cfg["multimodal"]["modalities"]) & set(["ECG", "EDA", "RR"])
                    )
                )
                > 0
            ):
                args = zip(
                    tamp_X,
                    tamp_ys,
                    tamp_ids,
                    repeat(cfg["multimodal"]["physio"]["window"]),
                    repeat(cfg["multimodal"]["physio"]["stride"]),
                )
                with Pool(
                    processes=cfg["multimodal"]["hyperparams"]["num_workers"]
                ) as pooling:
                    res = pooling.starmap(StressID.sliding, args)
                    X = np.concatenate(
                        [mts for mts, _, _ in res if mts is not None], axis=0
                    )
                    y = np.concatenate([y for _, y, _ in res if len(y) > 0], axis=0)
                    ids = np.concatenate(
                        [files for _, _, files in res if len(files) > 0], axis=0
                    )
                    ids, y, X = StressID.select_sequences(X, y, ids, self.ids)
                    print(X.shape)
                    self.physio = (ids, y, X)
        ### no sliding window ###
        else:
            logging.info("No sliding window for physio or audio.")
            if (
                len(
                    list(
                        set(cfg["multimodal"]["modalities"]) & set(["ECG", "EDA", "RR"])
                    )
                )
                > 0
            ):
                ids = np.asarray(
                    [f"{tamp_ids[i]}:0:{tamp_ys[i]}" for i in range(len(tamp_ids))]
                )
                ys = np.asarray(tamp_ys)
                self.physio = StressID.select_sequences(tamp_X, ys, ids, self.ids)
            if "AUDIO" in cfg["multimodal"]["modalities"]:
                ids = np.asarray(
                    [
                        f"{tamp_ids_audio[i]}:0:{tamps_ys_audio[i]}"
                        for i in range(len(tamp_ids_audio))
                    ]
                )
                ys = np.asarray(tamps_ys_audio)
                self.audio = StressID.select_sequences(tamp_audio, ys, ids, self.ids)
        return self

    @staticmethod
    def open_file(path):
        """open the features."""
        values, ids, labels = [], [], []
        with open(path, "r") as file:
            content = file.readlines()
        for line in content:
            line = line.split(":")
            values.append(np.asarray(line[5][1:-3].split(", "), dtype=np.float32))
            ids.append("_".join(line[0:3]))
            labels.append(float(line[3]))
        final_X = np.stack(values, axis=0)
        final_y = np.asarray(labels)
        final_files = np.asarray(ids)
        return final_X, final_y, final_files

    @staticmethod
    def get_videos(launches: np.array, labels: np.array, path_features: str):
        """
        Load txt.

        Parameters
        ----------
        filename : str
            Path to the file where the features are stored.
        launches : np.array
            List of videos to extract features from.
        """
        values, ids, ys = [], [], []
        with open(path_features, "r") as file:
            content = file.readlines()
        for line in content:
            line = line.split(":")
            values.append(np.asarray(line[5][1:-3].split(", "), dtype=np.float32))
            ids.append("_".join(line[0:3]))
            ys.append(float(line[3]))
        data = (np.stack(values, axis=0), np.asarray(ys), np.asarray(ids))
        video_ids = ["_".join(x[0:2]) for x in np.char.split(data[2], sep="_")]
        idx_to_keep = np.where(np.isin(video_ids, launches))[0]
        features = data[0][idx_to_keep]
        ids = data[2][idx_to_keep]
        # labelize here
        labels = np.asarray([labels["_".join(video.split("_")[:2])] for video in ids])
        final_ids = []
        for id, label in zip(ids, labels):
            final_ids.append(id.replace("_", ":") + ":" + str(label))
        ids = np.asarray(final_ids)
        return (ids, labels, features)

    @staticmethod
    def sliding(X, y, file_id, window, stride):
        """
        Create a sliding window for the data.

        Parameters
        ----------
        X : array
            array of the data.
        y : array
            array of the labels.
        window : int
            size of the window.
        stride : int
            stride of the window.
        """
        ## for each sample slide.
        X = np.asarray(X, dtype=np.float32)
        if X.shape[0] == 1 or X.ndim == 1:  # if only one chanel is selected
            X = X.squeeze()
            res = np.lib.stride_tricks.sliding_window_view(X, window)[::stride]
            res = np.expand_dims(res, axis=1)
        else:
            res = np.lib.stride_tricks.sliding_window_view(X, (X.shape[0], window))[
                0, ::stride
            ]
        y_slide = np.asarray([y] * res.shape[0], dtype=np.int32)
        id_slide = [f"{file_id}:{idx}:{y}" for idx in range(res.shape[0])]
        return res, y_slide, id_slide

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = {}
        ### for each modality (ids, y, X) ###
        matching_indices = self.indices[index]
        if "VIDEO" in self.cfg["multimodal"]["modalities"]:
            idx_video = matching_indices[self.mapping.index("VIDEO")]
            if idx_video is not None:
                item["id_VIDEO"] = self.video[0][idx_video]
                item["label"] = self.video[1][idx_video]
                item["VIDEO"] = torch.tensor(self.video[2][idx_video])
            else:
                item["VIDEO"] = torch.zeros(768)
                item["id_VIDEO"] = "missing"
        if "AUDIO" in self.cfg["multimodal"]["modalities"]:
            idx_audio = matching_indices[self.mapping.index("AUDIO")]
            if idx_audio is not None:
                item["AUDIO"] = audio_preprocess(
                    cfg=self.cfg["multimodal"], data=self.audio[2][idx_audio].squeeze()
                )
                item["id_AUDIO"] = self.audio[0][idx_audio]
                item["label"] = self.audio[1][idx_audio]
            else:
                if (
                    self.cfg["multimodal"]["audio_hyperparams"]["window_audio"]
                    == 480000
                ):
                    item["AUDIO"] = torch.unsqueeze(torch.zeros((64, 469)), 0)
                elif (
                    self.cfg["multimodal"]["audio_hyperparams"]["window_audio"]
                    == 960000
                ):
                    item["AUDIO"] = torch.unsqueeze(torch.zeros((64, 938)), 0)
                item["id_AUDIO"] = "missing"
        if "ECG" in self.cfg["multimodal"]["modalities"]:
            idx_physio = matching_indices[self.mapping.index("PHYSIO")]
            mod = self.physio_map.index("ECG")
            item["ECG"] = torch.tensor(
                self.physio[2][idx_physio][mod], dtype=torch.float32
            )
            item["ECG"] = (item["ECG"] - self.mean_physio[mod][0]) / self.std_physio[
                mod
            ][0]
            item["id_ECG"] = self.physio[0][idx_physio] + "_ECG"
            item["label"] = self.physio[1][idx_physio]
        if "EDA" in self.cfg["multimodal"]["modalities"]:
            idx_physio = matching_indices[self.mapping.index("PHYSIO")]
            mod = self.physio_map.index("EDA")
            item["EDA"] = torch.tensor(
                self.physio[2][idx_physio][mod], dtype=torch.float32
            )

            item["EDA"] = (item["EDA"] - self.mean_physio[mod][0]) / self.std_physio[
                mod
            ][0]
            item["id_EDA"] = self.physio[0][idx_physio] + "_EDA"
            item["label"] = self.physio[1][idx_physio]
        if "RR" in self.cfg["multimodal"]["modalities"]:
            idx_physio = matching_indices[self.mapping.index("PHYSIO")]
            mod = self.physio_map.index("RR")
            item["RR"] = torch.tensor(
                self.physio[2][idx_physio][mod], dtype=torch.float32
            )
            item["RR"] = (item["RR"] - self.mean_physio[mod][0]) / self.std_physio[mod][
                0
            ]
            item["id_RR"] = self.physio[0][idx_physio] + "_RR"
            item["label"] = self.physio[1][idx_physio]

        if self.stage == "contrastive":
            item = self.augmentations(item)
        else:
            binarize = torch.zeros(2, dtype=torch.float32)
            binarize[item["label"]] = 1
            item["label"] = binarize
        return item


# import os
# import logging
# import pyrootutils
# from multiprocessing import Pool
# from copy import deepcopy
# import torch
# import numpy as np
# import pandas as pd
# import torch.utils.data
# from itertools import repeat
# import librosa


# from src.utils.preprocessing.audio_preprocessing import audio_preprocess
# from src.utils.training.augmentations import MultiModalAugmentations


# class StressID(torch.utils.data.Dataset):
#     def __init__(self, cfg, ids, stage="anchor") -> None:
#         """Initialize the StressID dataset

#         Parameters
#         ----------
#         cfg : dict
#             Configuration dictionary
#         stage : str, optional
#             Stage of the dataset, by default "anchor"
#         split : str, optional
#             Split of the dataset, by default "train"
#         """
#         super().__init__()
#         self.cfg = cfg
#         self.ids = ids
#         self.stage = stage
#         self.mask = None
#         self.physio, self.video, self.audio = None, None, None
#         labels = pd.read_csv(
#             os.path.join(cfg["paths"]["data"], cfg["multimodal"]["path_labels"]),
#             sep=",",
#             header=0,
#             index_col=0,
#         ).dropna()["binary-stress"]
#         modality_ids, self.mapping = [], []
#         self.physio_map = deepcopy(cfg["multimodal"]["modalities"])
#         self.get_physio_audio(labels, cfg)
#         if (
#             len(
#                 list(
#                     set(cfg["multimodal"]["modalities"])
#                     & set(["ECG", "EDA", "RR", "AUDIO"])
#                 )
#             )
#             > 0
#         ):
#             if (
#                 len(
#                     list(
#                         set(cfg["multimodal"]["modalities"]) & set(["ECG", "EDA", "RR"])
#                     )
#                 )
#                 > 0
#             ):
#                 self.mapping.append("PHYSIO")
#                 modality_ids.append([self.physio[0].tolist(), self.physio[1]])
#             if "AUDIO" in cfg["multimodal"]["modalities"]:
#                 modality_ids.append([self.audio[0].tolist(), self.audio[1]])
#                 self.mapping.append("AUDIO")
#                 self.physio_map.remove("AUDIO")
#         if "VIDEO" in cfg["multimodal"]["modalities"]:
#             if self.cfg["multimodal"]["video"]["precomputed_video"]:
#                 self.video = StressID.get_videos(
#                     self.ids,
#                     labels,
#                     os.path.join(
#                         cfg["paths"]["data"], cfg["multimodal"]["path_features_video"]
#                     ),
#                 )
#                 self.mapping.append("VIDEO")
#                 modality_ids.append([self.video[0].tolist(), self.video[1]])
#                 self.physio_map.remove("VIDEO")

#         ### merge ###
#         keep_missing = (
#             False
#             if (stage in ["anchor", "contrastive"])
#             else self.cfg["multimodal"]["keep_missing"]
#         )
#         if len(self.mapping) > 1:
#             self.indices, self.labels = self.merge(
#                 modality_ids, keep_missing=keep_missing
#             )
#         else:
#             self.indices = modality_ids[0][0]
#             self.labels = labels

#         if self.stage == "contrastive":
#             self.augmentations = MultiModalAugmentations(
#                 modalities=self.cfg["multimodal"]["modalities"],
#                 modality_dropout=self.cfg["model"]["contrastive_loss"][
#                     "modality_dropout"
#                 ],
#                 noise_sigma=self.cfg["model"]["contrastive_loss"]["noise_sigma"],
#             )

#     @staticmethod
#     def merge(lists: list, keep_missing=False):
#         """
#         Merge the different modalities

#         Parameters
#         ----------
#         lists : list
#             List of modalities ids
#         keep_missing : bool, optional
#             Keep missing values, by default False

#         Returns
#         -------
#         tuple : (np.ndarray, np.array)
#             Indices, labels
#         """
#         sets = [set(lst[0]) for lst in lists]
#         all_values = set.union(*sets)
#         result, labels = [], []
#         for val in all_values:
#             all_in = all([val in lst[0] for lst in lists])
#             if all_in or keep_missing:
#                 indices = tuple(
#                     sublist[0].index(val) if val in sublist[0] else None
#                     for sublist in lists
#                 )
#                 labels.append(
#                     [
#                         lists[idx_list][1][indice]
#                         for idx_list, indice in enumerate(indices)
#                         if indice is not None
#                     ][0]
#                 )
#                 result.append(indices)
#         return result, labels

#     @staticmethod
#     def select_sequences(X, y, files: np.ndarray, files_to_keep: list):
#         """
#         Select the sequences to keep

#         Parameters
#         ----------
#         X : np.ndarray or list
#             Data
#         y : np.array
#             Labels
#         files : np.ndarray
#             Files
#         files_to_keep : list
#             Files to keep

#         Returns
#         -------
#         tuple : (np.ndarray, np.array, iterable)
#             Files, labels, data
#         """
#         ids = np.array(["_".join(id.split(":")[0:2]) for id in files])
#         selected = np.where(np.isin(ids, files_to_keep))[0]
#         X = X[selected] if isinstance(X, np.ndarray) else [X[i] for i in selected]
#         y = y[selected]
#         files = files[selected]
#         return (files, y, X)

#     def get_physio_audio(self, labels, cfg):
#         """Get the physiological data

#         Parameters
#         ----------
#         labels : pd.DataFrame
#             Labels of the data
#         cfg : dict
#             Configuration dictionary
#         """
#         path_physio = os.path.join(
#             cfg["paths"]["data"], cfg["multimodal"]["physiological_root"]
#         )
#         path_audio = os.path.join(cfg["paths"]["data"], cfg["multimodal"]["audio_root"])
#         files = [
#             f
#             for f_sub in [f_names for _, _, f_names in os.walk(path_physio)]
#             for f in f_sub
#         ]
#         files_audio = [
#             f
#             for f_sub in [f_names for _, _, f_names in os.walk(path_audio)]
#             for f in f_sub
#         ]
#         tamp_X, tamp_ids, tamp_ys, tamp_audio, tamp_ids_audio, tamps_ys_audio = (
#             [],
#             [],
#             [],
#             [],
#             [],
#             [],
#         )
#         ### extract physio & audio from files ###
#         for file in files:
#             if not file.startswith("."):
#                 id_file = file.split(".")[0]
#                 tamp = []
#                 if id_file in labels.index.values.tolist():
#                     tamp_ids.append(id_file.replace("_", ":"))
#                     tamp_ys.append(labels.loc[id_file])
#                     path = os.path.join(
#                         path_physio, file.split("_")[0], file  # num of the folder
#                     )
#                     file = pd.read_csv(path, sep=",")
#                     if "ECG" in cfg["multimodal"]["modalities"]:
#                         tamp.append(file["ECG"].values.tolist())
#                     if "EDA" in cfg["multimodal"]["modalities"]:
#                         tamp.append(file["EDA"].values.tolist())
#                     if "RR" in cfg["multimodal"]["modalities"]:
#                         tamp.append(file["RR"].values.tolist())
#                     if len(tamp) > 0:
#                         tamp_X.append(tamp)
#         #### AUDIO ####
#         if "AUDIO" in cfg["multimodal"]["modalities"]:
#             ### to complete ###
#             for f in files_audio:
#                 if not f.startswith("."):
#                     id_file = f.split(".")[0]
#                     if id_file in labels.index.values.tolist():
#                         tamp_ids_audio.append(id_file.replace("_", ":"))
#                         tamps_ys_audio.append(labels.loc[id_file])
#                         ## did not read the file ??
#                         folder = f.split("_")[0]
#                         sound, _ = librosa.load(
#                             os.path.join(path_audio, folder, f), sr=None
#                         )
#                         tamp_audio.append(sound)

#         ### sliding window - if necessary ###
#         if cfg["multimodal"]["hyperparams"]["sliding_window"]:
#             if "AUDIO" in cfg["multimodal"]["modalities"]:
#                 args = zip(
#                     tamp_audio,
#                     tamps_ys_audio,
#                     tamp_ids_audio,
#                     repeat(cfg["multimodal"]["audio_hyperparams"]["window_audio"]),
#                     repeat(cfg["multimodal"]["audio_hyperparams"]["stride_audio"]),
#                 )
#                 with Pool(
#                     processes=cfg["multimodal"]["hyperparams"]["num_workers"]
#                 ) as pooling:
#                     res = pooling.starmap(StressID.sliding, args)
#                     X_audio = np.concatenate(
#                         [mts for mts, _, _ in res if mts is not None], axis=0
#                     )
#                     y = np.concatenate([y for _, y, _ in res if len(y) > 0], axis=0)
#                     ids = np.concatenate(
#                         [files for _, _, files in res if len(files) > 0], axis=0
#                     )
#                     ids, y, X_audio = StressID.select_sequences(
#                         X_audio, y, ids, self.ids
#                     )
#                     self.audio = (ids, y, X_audio)
#             if (
#                 len(
#                     list(
#                         set(cfg["multimodal"]["modalities"]) & set(["ECG", "EDA", "RR"])
#                     )
#                 )
#                 > 0
#             ):
#                 args = zip(
#                     tamp_X,
#                     tamp_ys,
#                     tamp_ids,
#                     repeat(cfg["multimodal"]["physio"]["window"]),
#                     repeat(cfg["multimodal"]["physio"]["stride"]),
#                 )
#                 with Pool(
#                     processes=cfg["multimodal"]["hyperparams"]["num_workers"]
#                 ) as pooling:
#                     res = pooling.starmap(StressID.sliding, args)
#                     X = np.concatenate(
#                         [mts for mts, _, _ in res if mts is not None], axis=0
#                     )
#                     y = np.concatenate([y for _, y, _ in res if len(y) > 0], axis=0)
#                     ids = np.concatenate(
#                         [files for _, _, files in res if len(files) > 0], axis=0
#                     )
#                     ids, y, X = StressID.select_sequences(X, y, ids, self.ids)
#                     self.physio = (ids, y, X)
#         ### no sliding window ###
#         else:
#             logging.info("No sliding window for physio or audio.")
#             if (
#                 len(
#                     list(
#                         set(cfg["multimodal"]["modalities"]) & set(["ECG", "EDA", "RR"])
#                     )
#                 )
#                 > 0
#             ):
#                 ids = np.asarray(
#                     [f"{tamp_ids[i]}:0:{tamp_ys[i]}" for i in range(len(tamp_ids))]
#                 )
#                 ys = np.asarray(tamp_ys)
#                 self.physio = StressID.select_sequences(tamp_X, ys, ids, self.ids)
#             if "AUDIO" in cfg["multimodal"]["modalities"]:
#                 ids = np.asarray(
#                     [
#                         f"{tamp_ids_audio[i]}:0:{tamps_ys_audio[i]}"
#                         for i in range(len(tamp_ids_audio))
#                     ]
#                 )
#                 ys = np.asarray(tamps_ys_audio)
#                 self.audio = StressID.select_sequences(tamp_audio, ys, ids, self.ids)
#         return self

#     @staticmethod
#     def open_file(path):
#         values, ids, labels = [], [], []
#         with open(path, "r") as file:
#             content = file.readlines()
#         for line in content:
#             line = line.split(":")
#             values.append(np.asarray(line[5][1:-3].split(", "), dtype=np.float32))
#             ids.append("_".join(line[0:3]))
#             labels.append(float(line[3]))
#         final_X = np.stack(values, axis=0)
#         final_y = np.asarray(labels)
#         final_files = np.asarray(ids)
#         return final_X, final_y, final_files

#     @staticmethod
#     def get_videos(launches: np.array, labels: np.array, path_features: str):
#         """
#         Load txt.

#         Parameters
#         ----------
#         filename : str
#             Path to the file where the features are stored.
#         launches : np.array
#             List of videos to extract features from.
#         """
#         values, ids, ys = [], [], []
#         with open(path_features, "r") as file:
#             content = file.readlines()
#         for line in content:
#             line = line.split(":")
#             values.append(np.asarray(line[5][1:-3].split(", "), dtype=np.float32))
#             ids.append("_".join(line[0:3]))
#             ys.append(float(line[3]))
#         data = (np.stack(values, axis=0), np.asarray(ys), np.asarray(ids))
#         video_ids = ["_".join(x[0:2]) for x in np.char.split(data[2], sep="_")]
#         idx_to_keep = np.where(np.isin(video_ids, launches))[0]
#         features = data[0][idx_to_keep]
#         ids = data[2][idx_to_keep]
#         # labelize here
#         labels = np.asarray([labels["_".join(video.split("_")[:2])] for video in ids])
#         final_ids = []
#         for id, label in zip(ids, labels):
#             final_ids.append(id.replace("_", ":") + ":" + str(label))
#         ids = np.asarray(final_ids)
#         return (ids, labels, features)

#     @staticmethod
#     def sliding(X, y, file_id, window, stride):
#         """
#         Create a sliding window for the data.

#         Parameters
#         ----------
#         X : array
#             array of the data.
#         y : array
#             array of the labels.
#         window : int
#             size of the window.
#         stride : int
#             stride of the window.
#         """
#         ## for each sample slide.
#         X = np.asarray(X, dtype=np.float32)
#         if X.shape[0] == 1 or X.ndim == 1:  # if only one chanel is selected
#             X = X.squeeze()
#             res = np.lib.stride_tricks.sliding_window_view(X, window)[::stride]
#             res = np.expand_dims(res, axis=1)
#         else:
#             res = np.lib.stride_tricks.sliding_window_view(X, (X.shape[0], window))[
#                 0, ::stride
#             ]
#         y_slide = np.asarray([y] * res.shape[0], dtype=np.int32)
#         id_slide = [f"{file_id}:{idx}:{y}" for idx in range(res.shape[0])]
#         return res, y_slide, id_slide

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, index):
#         item = {}
#         ### for each modality (ids, y, X) ###
#         matching_indices = self.indices[index]
#         if "VIDEO" in self.cfg["multimodal"]["modalities"]:
#             idx_video = matching_indices[self.mapping.index("VIDEO")]
#             if idx_video is not None:
#                 item["id_VIDEO"] = self.video[0][idx_video]
#                 item["label"] = self.video[1][idx_video]
#                 item["VIDEO"] = torch.tensor(self.video[2][idx_video])
#             else:
#                 item["VIDEO"] = torch.zeros(768)
#                 item["id_VIDEO"] = "missing"
#         if "AUDIO" in self.cfg["multimodal"]["modalities"]:
#             idx_audio = matching_indices[self.mapping.index("AUDIO")]
#             if idx_audio is not None:
#                 item["AUDIO"] = audio_preprocess(
#                     cfg=self.cfg["multimodal"], data=self.audio[2][idx_audio].squeeze()
#                 )
#                 item["id_AUDIO"] = self.audio[0][idx_audio]
#                 item["label"] = self.audio[1][idx_audio]
#             else:
#                 if (
#                     self.cfg["multimodal"]["audio_hyperparams"]["window_audio"]
#                     == 480000
#                 ):
#                     item["AUDIO"] = torch.unsqueeze(torch.zeros((64, 469)), 0)
#                 elif (
#                     self.cfg["multimodal"]["audio_hyperparams"]["window_audio"]
#                     == 960000
#                 ):
#                     item["AUDIO"] = torch.unsqueeze(torch.zeros((64, 938)), 0)
#                 item["id_AUDIO"] = "missing"
#         if "ECG" in self.cfg["multimodal"]["modalities"]:
#             idx_physio = matching_indices[self.mapping.index("PHYSIO")]
#             mod = self.physio_map.index("ECG")
#             item["ECG"] = torch.tensor(
#                 self.physio[2][idx_physio][mod], dtype=torch.float32
#             )
#             item["id_ECG"] = self.physio[0][idx_physio] + "_ECG"
#             item["label"] = self.physio[1][idx_physio]
#         if "EDA" in self.cfg["multimodal"]["modalities"]:
#             idx_physio = matching_indices[self.mapping.index("PHYSIO")]
#             mod = self.physio_map.index("EDA")
#             item["EDA"] = torch.tensor(
#                 self.physio[2][idx_physio][mod], dtype=torch.float32
#             )
#             item["id_EDA"] = self.physio[0][idx_physio] + "_EDA"
#             item["label"] = self.physio[1][idx_physio]
#         if "RR" in self.cfg["multimodal"]["modalities"]:
#             idx_physio = matching_indices[self.mapping.index("PHYSIO")]
#             mod = self.physio_map.index("RR")
#             item["RR"] = torch.tensor(
#                 self.physio[2][idx_physio][mod], dtype=torch.float32
#             )
#             item["id_RR"] = self.physio[0][idx_physio] + "_RR"
#             item["label"] = self.physio[1][idx_physio]
#         if self.stage == "contrastive":
#             item = self.augmentations(item)
#         else:
#             binarize = torch.zeros(2, dtype=torch.float32)
#             binarize[item["label"]] = 1
#             item["label"] = binarize
#         return item
