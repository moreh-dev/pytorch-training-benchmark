import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import os


class HFIterableDataset(IterableDataset):

    def __init__(self, hf_dataset):
        super().__init__()
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        print(self.dataset)
        print(len(self.dataset))

        for item in self.dataset:
            #print(item)
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
            labels = torch.tensor(item["labels"], dtype=torch.long)
            yield {"input_ids": input_ids, "labels": labels}


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    for i in range(len(input_ids)):
        print(
            f"[i={i}] {input_ids[i].shape}, {input_ids[i].dtype}, {labels[i].shape}, {labels[i].dtype}"
        )

    input_ids_padded = pad_sequence(input_ids,
                                    batch_first=True,
                                    padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids_padded, "labels": labels_padded}


if __name__ == '__main__':
    dataset_dir = "/vast/huggingface/hub/datasets--regisss--scrolls_gov_report_preprocessed_mlperf_2/snapshots/21ff1233ee3e87bc780ab719c755170148aba1cb/data"
    train_dataset_path = os.path.join(dataset_dir,
                                      "train-00000-of-00001.parquet")

    hf_dataset = load_dataset("parquet",
                              data_files=train_dataset_path,
                              split="train")
    torch_dataset = HFIterableDataset(hf_dataset)

    data_loader = DataLoader(torch_dataset,
                             batch_size=4,
                             collate_fn=collate_fn)

    for index, batch in enumerate(data_loader):
        print(f'{index}, {batch["input_ids"].shape}, {batch["labels"].shape}')
