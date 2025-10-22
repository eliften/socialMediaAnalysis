import pandas as pd, yaml, logging
from pathlib import Path
from loguru import logger

class DataLoader:
    def __init__(self, config_path: str = r"configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.data_config = self.config['data']

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def load_topics(self, clean=False):
        if clean:
            path = Path("data") / 'topics_clean.csv'
        else:
            path = Path(self.data_config['topics_file'])
        try:
            topics = pd.read_csv(path)
            logger.info(f"Topics data loaded successfully from {path}")
            return pd.DataFrame(topics)
        except Exception as e:
            logger.error(f"Error loading topics data: {e}")
            raise

    def load_conclusions(self, clean:False):
        if clean:
            path = Path("data") / 'conclusions_clean.csv'
        else:
            path = Path(self.data_config['conclusions_file'])
        try:
            conclusions = pd.read_csv(path)
            logger.info(f"Conclusions data loaded successfully from {path}")
            return pd.DataFrame(conclusions)
        except Exception as e:
            logger.error(f"Error loading conclusions data: {e}")
            raise

    def load_opinions(self, clean=False):
        if clean:
            path = Path("data") / 'opinions_clean.csv'
        else:
            path = Path(self.data_config['opinions_file'])
        try:
            opinions = pd.read_csv(path)
            logger.info(f"Opinions data loaded successfully from {path}")
            return pd.DataFrame(opinions)
        except Exception as e:
            logger.error(f"Error loading opinions data: {e}")
            raise

    def load_all_data(self, clean=False):
        topics = self.load_topics(clean)
        conclusions = self.load_conclusions(clean)
        opinions = self.load_opinions(clean)
        return topics, conclusions, opinions
