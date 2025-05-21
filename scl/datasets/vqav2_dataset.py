from .base_dataset import BaseDataset


class VQAv2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqav2_train", "vqav2_trainable_val"] # This training data is consistent with ViLT
            # names = ["vqav2_train", "vqav2_trainable_val", "vgqa_train"] # This training data is consistent with UNITER, ALBEF
        elif split == "val":
            names = ["vqav2_rest_val"]
        elif split == "test":
            names = ["vqav2_test"]  # vqav2_test-dev for test-dev

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

        if split == "test":
            self._truncate_test_samples()
        
        self.max_test_samples = 100

    def _truncate_test_samples(self):
        if hasattr(self, "index_mapper"):
            if isinstance(self.index_mapper, dict):
                # Lấy 100 phần tử đầu tiên
                items = list(self.index_mapper.items())[:self.max_test_samples]
                self.index_mapper = dict(items)
            else:
                # Nếu là list hoặc tuple
                self.index_mapper = self.index_mapper[:self.max_test_samples]
        if hasattr(self, "table") and hasattr(self.table, "__len__"):
            if len(self.table) > self.max_test_samples:
                self.table = self.table.slice(0, self.max_test_samples)

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        if self.split != "test":
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["answer_labels"][index][question_index].as_py()
            scores = self.table["answer_scores"][index][question_index].as_py()
        else:
            answers = list()
            labels = list()
            scores = list()

        return {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
        }
