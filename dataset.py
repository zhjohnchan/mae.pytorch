from pathlib import Path
from typing import Optional

import datasets
import pyarrow as pa
from dataclasses import dataclass
from datasets.tasks import ImageClassification

logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class ImageFolderConfig(datasets.BuilderConfig):
    """BuilderConfig for ImageFolder."""
    features: Optional[datasets.Features] = None

    @property
    def schema(self):
        return pa.schema(self.features.type) if self.features is not None else None


class ImageFolder(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = ImageFolderConfig

    def _info(self):
        folder = None
        if isinstance(self.config.data_dir, str):
            folder = self.config.data_dir
        elif isinstance(self.config.data_dir, dict):
            folder = self.config.data_dir.get('train', None)
        if folder is None:
            raise RuntimeError()
        classes = sorted([x.name.lower() for x in Path(folder).glob('*/**')])
        return datasets.DatasetInfo(
            features=datasets.Features(
                {"image_file_path": datasets.Value("string"),
                 "labels": datasets.features.ClassLabel(names=classes),}
            ),
            task_templates=[
                datasets.tasks.ImageClassification(image_file_path_column="image_file_path", label_column="labels",
                                                   labels=classes)]
        )

    def _split_generators(self, dl_manager):
        if not self.config.data_dir:
            raise ValueError(f"At least one data file must be specified, but got data_dir={self.config.data_dir}")
        data_dir = self.config.data_dir
        if isinstance(data_dir, str):
            folder = data_dir
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"archive_path": folder})]
        splits = []
        for split_name, folder in data_dir.items():
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"archive_path": folder}))
        return splits

    def _generate_examples(self, archive_path):
        logger.info("generating examples from = %s", archive_path)
        extensions = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".JPEG"}
        for i, path in enumerate(Path(archive_path).glob("**/*")):
            if path.suffix in extensions:
                yield i, {"image_file_path": path.as_posix(), "labels": path.parent.name.lower()}
