import dill
import os
from typing import Dict
from pathlib import Path


class BinaryStorage:
    def __init__(self, file_path, default_data):
        """Initialise the storage with the specified file path"""
        self.file_path = file_path
        self.data: Dict = default_data
        # Create directories if they do not exist
        self._ensure_path()
        # Load data from the file if it exists
        if os.path.exists(file_path):
            self._load(default_data)
        
        # Create a file with `default_data` if it does not exist
        self.dump()
    
    # def get(self, key: Any, default=None) -> Any | None:
    #     data = self.data.get(key)
    #     return data if data is not None else default
    
    def _ensure_path(self):
        """Create all directories for the file path"""
        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    def _load(self, default_data):
        try:
            with open(self.file_path, 'rb') as f:
                self.data = dill.load(f)
        
        except (FileNotFoundError, dill.PickleError) as e:
            print(f"Error whilst loading data: {e}")
            self.data = default_data
    
    def dump(self):
        try:
            if Path(self.file_path).exists():
                with open(self.file_path, 'wb') as f:
                    dill.dump(self.data, f)
            else:
                self._ensure_path()
                Path(self.file_path).touch(exist_ok=True)
                with open(self.file_path, 'wb') as f:
                    dill.dump(self.data, f)
        except (IOError, dill.PickleError) as e:
            print(f"Error whilst saving data: {e}")
    
    def __enter__(self):
        """Enter the context"""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context: automatic dump"""
        self.dump()