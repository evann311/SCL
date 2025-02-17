import json
import os
import pandas as pd
import pyarrow as pa
import random
import gc

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(path, iid2captions):
    split = 'train'
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]

    return [binary, captions, name, split]


def make_arrow(root, dataset_root):
    with open(f"{root}/vgqa_annt.json", "r") as fp:
        captions = json.load(fp)

    iid2captions = defaultdict(list)

    for cap in tqdm(captions):
        iid = cap[0].split('/')[-1]
        iid2captions[iid] = [cap[1]]

    paths = list(glob(f"{root}/images/*.jpg"))
    random.shuffle(paths)
    caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    sub_len = int(len(caption_paths) // 100000)
    subs = list(range(sub_len + 1))

    for sub in subs:
        sub_paths = caption_paths[sub * 100000: (sub + 1) * 100000]
        bs = [path2rest(path, iid2captions) for path in tqdm(sub_paths)]
        dataframe = pd.DataFrame(
            bs, columns=["image", "caption", "image_id", "split"],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
                f"{dataset_root}/pretrain_vg_{sub}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        del dataframe
        del table
        del bs
        gc.collect()
