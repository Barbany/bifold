from torch.utils.data import Dataset as TorchDataset

from .processor import Processor


class BaseDataset(TorchDataset):
    def __init__(self, cfg, processor_config, partition="train", *args, **kwargs):
        super().__init__()

        assert partition in ["train", "test"]
        self.partition = partition

        self.cfg = cfg
        self.dataset_path = cfg.dataset_path
        self.depth_scale = cfg.depth_scale

        self.processor = Processor(
            cfg=processor_config,
            partition=partition,
            num_nodes=cfg.num_nodes,
            neighbor_radius=cfg.neighbor_radius,
            voxel_size=cfg.voxel_size,
            *args,
            **kwargs,
        )


class Datasets:
    @staticmethod
    def get_by_name(cfg, partition, *args, **kwargs) -> BaseDataset:
        if cfg.name == "bimanual":
            from .bimanual_dataset import BimanualDataset as Dataset
        elif cfg.name == "bimanual_sequential":
            from .bimanual_dataset_sequential import BimanualDatasetSequential as Dataset
        elif cfg.name == "single":
            from .single_dataset import SingleDataset as Dataset
        elif cfg.name == "single_sequential":
            from .single_dataset_sequential import SingleDatasetSequential as Dataset
        elif cfg.name == "real":
            from .real_dataset import RealDataset as Dataset
        else:
            raise ValueError(f"Dataset {cfg.name} not recognized")
        return Dataset(cfg, *args, **kwargs, partition=partition)

    @staticmethod
    def get_dataloaders(cfg, *args, **kwargs):
        if cfg.processor.requires_graph:
            from torch_geometric.loader import DataLoader
        else:
            from torch.utils.data import DataLoader

        if cfg.eval_only:
            train_dataloader = None
        else:
            train_dataset = Datasets.get_by_name(
                cfg.train_dataset,
                *args,
                **kwargs,
                processor_config=cfg.processor,
                partition="train",
                autoprocessor_name=cfg["model"].get("automodel_name"),
            )

            if cfg.debug:
                train_dataset[0]

            train_dataloader = DataLoader(
                dataset=train_dataset,  # type: ignore
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
            )

        if cfg.test_dataset.name is None:
            cfg.test_dataset = cfg.train_dataset

        test_dataset = Datasets.get_by_name(
            cfg.test_dataset,
            *args,
            **kwargs,
            processor_config=cfg.processor,
            partition="test",
            autoprocessor_name=cfg["model"].get("automodel_name"),
        )

        if cfg.debug:
            test_dataset[0]

        input_processor = test_dataset.processor

        test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=cfg.test_batch_size  # type: ignore
        )

        return train_dataloader, test_dataloader, input_processor
