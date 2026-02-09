import lightning as L
import litdata as ld


class DefaultDataset(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.index_path = config.index_path
        self.batch_size = config.batch_size
        self.cache_dir = config.cache_dir

    def train_dataloader(self):
        return ld.StreamingDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn_train,
            num_workers=8,
            shuffle=True,
        )

    def val_dataloader(self):
        return ld.StreamingDataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn_val,
            num_workers=8,
        )

    def collate_fn_train(self, batch):
        return self.collate(batch, "train")

    def collate_fn_val(self, batch):
        return self.collate(batch, "val")
