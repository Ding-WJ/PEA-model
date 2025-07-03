

import csv
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from typing import Tuple, Dict, Optional

class ProteomicsDataset(Dataset):
    def __init__(self, csv_file: str):
        """
        Initializes the dataset by loading data from the CSV file.

        Args:
            csv_file (str): Path to the input CSV file.
        """
        self.patient_data: Dict[str, Dict[str, list]] = {}
        self.protein_dimensions: Dict[str, int] = {}

        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)

        # Extract patient IDs and protein names
        patient_ids = rows[0][1:]  # First row (excluding the first column)
        protein_names = [row[0] for row in rows[1:]]

        # Map protein names to their indices
        for idx, protein_name in enumerate(protein_names):
            self.protein_dimensions[protein_name] = idx

        # Extract patient data and labels
        for col_idx, patient_id in enumerate(patient_ids):
            # Representation: protein expression levels
            representation = [
                float(row[col_idx + 1]) if row[col_idx + 1] not in ['NA', ''] else 0.0
                for row in rows[1:-1]  # Exclude last row (labels)
            ]
            # Label: last row
            label = float(rows[-1][col_idx + 1]) if rows[-1][col_idx + 1] not in ['NA', ''] else 0.0

            self.patient_data[patient_id] = {
                'representation': representation,
                'label': label
            }

    def __len__(self) -> int:
        return len(self.patient_data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        patient_id = list(self.patient_data.keys())[idx]
        patient_data = self.patient_data[patient_id]
        representation = np.array(patient_data['representation'], dtype=np.float32)
        label = np.array(patient_data['label'], dtype=np.int64)
        return representation, label

    def get_patient_data(self, patient_id: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        Retrieve representation and label for a specific patient.

        Args:
            patient_id (str): The ID of the patient.

        Returns:
            tuple: (representation, label) or None if patient not found
        """
        if patient_id in self.patient_data:
            data = self.patient_data[patient_id]
            representation = np.array(data['representation'], dtype=np.float32)
            label = int(data['label'])
            return representation, label
        else:
            print(f"Patient {patient_id} not found.")
            return None

    def save_protein_dimensions(self, output_file: str):
        """
        Save protein dimensions to a pickle (.pkl) file.

        Args:
            output_file (str): Path to save the protein dimensions.
        """
        with open(output_file, 'wb') as file:
            pickle.dump(self.protein_dimensions, file)

    def get_protein_dimensions(self) -> Dict[str, int]:
        """
        Return all protein IDs, their names, and corresponding row indices.

        Returns:
            dict: A dictionary with protein IDs as keys and row indices as values.
        """
        return self.protein_dimensions

def Proteomics_dataloaders(dataset: ProteomicsDataset, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                            batch_size: int = 32, shuffle: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders from the dataset.

    Args:
        dataset (ProteomicsDataset): The dataset instance.
        train_ratio (float): The ratio of training data (default is 0.8).
        val_ratio (float): The ratio of validation data (default is 0.1).
        batch_size (int): The batch size for DataLoader.
        shuffle (bool): Whether to shuffle the dataset before splitting.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    if shuffle:
        train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])
    else:
        indices = torch.arange(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
from typing import Tuple, Union
from torch.utils.data import DataLoader, Subset, random_split
import torch

def Proteomics_dataloaders8_2(dataset: ProteomicsDataset, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                            batch_size: int = 32, shuffle: bool = True
                           ) -> Tuple[DataLoader, Union[DataLoader, None], DataLoader]:
    """
    Create train, validation (optional), and test DataLoaders from the dataset.

    Args:
        dataset (ProteomicsDataset): The dataset instance.
        train_ratio (float): The ratio of training data.
        val_ratio (float): The ratio of validation data.
        batch_size (int): The batch size for DataLoader.
        shuffle (bool): Whether to shuffle the dataset before splitting.

    Returns:
        tuple: (train_loader, val_loader or None, test_loader)
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    if shuffle:
        subsets = random_split(dataset, [train_size, val_size, test_size])
        train_subset = subsets[0]
        val_subset = subsets[1] if val_size > 0 else None
        test_subset = subsets[2]
    else:
        indices = torch.arange(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size] if val_size > 0 else []
        test_indices = indices[train_size + val_size:]
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices) if val_size > 0 else None
        test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True) if val_subset is not None else None
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    dataset = ProteomicsDataset('/home/lixiaoyang/hello/data/G_DATA.csv')
    dataset.save_protein_dimensions('protein_dimensions_G_X.pkl')

    patient_id = list(dataset.patient_data.keys())[0]  # Example patient ID
    representation, label = dataset.get_patient_data(patient_id)

    print(f"Patient ID: {patient_id}")
    print(f"Representation: {representation}, Representation size: {len(representation)}")
    print(f"Label: {label}")

    train_loader, val_loader, test_loader = Proteomics_dataloaders(dataset, train_ratio=0.8, val_ratio=0.1, batch_size=32, shuffle=True)

    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(val_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")