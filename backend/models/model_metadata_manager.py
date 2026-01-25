#!/usr/bin/env python3
"""
Model Metadata Manager

A utility script to view and update metadata in joblib model files.
Supports adding/updating model information like name, id, type, etc.

Usage:
    python model_metadata_manager.py

The script will prompt for the model path and allow interactive metadata management.
"""

# Custom classes needed for unpickling models - must be defined before any imports
class StackingEnsemble:
    """Stacking ensemble class - must match training script for unpickling."""
    def __init__(self, bases, meta):
        self.bases = bases
        self.meta = meta
        self.classes_ = bases[0].classes_

    def predict_proba(self, X_scaled):
        import numpy as np
        base_probs = [b.predict_proba(X_scaled)[:, 1] for b in self.bases]
        stacked = np.column_stack(base_probs)
        meta_probs = self.meta.predict_proba(stacked)[:, 1]
        return np.column_stack([1 - meta_probs, meta_probs])

    def __repr__(self):
        """Return a more informative string representation."""
        base_names = [type(base).__name__ for base in self.bases]
        meta_name = type(self.meta).__name__
        return f"StackingEnsemble(bases=[{', '.join(base_names)}], meta={meta_name})"

import os
import sys
import joblib
from datetime import datetime
import json
import numpy as np

# Add model controllers and benchmarkers to path for custom classes
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'Model_Controllers'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'models', 'BC_Model', 'BC_Diagnostic_Model'))

print("✓ Loaded StackingEnsemble class")

def load_model_data(model_path):
    """Load model data from joblib file"""
    try:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return None

        print(f"Loading model from: {model_path}")
        model_data = joblib.load(model_path)

        # Handle different model storage formats
        if isinstance(model_data, dict):
            print(f"Successfully loaded model with keys: {list(model_data.keys())}")
        elif hasattr(model_data, 'predict'):  # Single model object
            print(f"Successfully loaded single model object: {type(model_data).__name__}")
            # Convert to dictionary format for consistency
            model_data = {'model': model_data}
            print("Converted to dictionary format with 'model' key")
        else:
            print(f"Loaded object type: {type(model_data).__name__}")
            # Try to wrap in dictionary anyway
            model_data = {'data': model_data}

        return model_data

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def display_metadata(model_data):
    """Display current metadata in the model"""
    print("\n" + "="*60)
    print("CURRENT METADATA")
    print("="*60)

    if 'metadata' not in model_data:
        print("No metadata found in model file.")
        print("💡 Tip: Add metadata to organize and document your models!")
        return

    metadata = model_data['metadata']
    if not metadata:
        print("Metadata dictionary exists but is empty.")
        return

    for key, value in metadata.items():
        print(f"{key}: {value}")

def inspect_model_keys(model_data):
    """Inspect all keys in the model data with detailed information"""
    print("\n" + "="*80)
    print("MODEL FILE INSPECTION - ALL KEYS")
    print("="*80)

    # Check if this was originally a single model object
    if len(model_data) == 1 and 'model' in model_data and 'metadata' not in model_data:
        print("📝 Note: This model was originally saved as a single object and has been")
        print("         wrapped in a dictionary for editing. The original model is under 'model' key.")
        print()

    for key, value in model_data.items():
        print(f"\n🔑 Key: '{key}'")
        print(f"   Type: {type(value).__name__}")

        # Handle different types of values
        if hasattr(value, '__len__') and not isinstance(value, str):
            try:
                if hasattr(value, 'shape'):  # NumPy arrays
                    print(f"   Shape: {value.shape}")
                    print(f"   Dtype: {value.dtype}")
                elif hasattr(value, 'n_features_in_'):  # scikit-learn objects
                    print(f"   Features: {value.n_features_in_}")
                    if hasattr(value, 'classes_'):
                        print(f"   Classes: {value.classes_}")
                elif isinstance(value, dict):
                    print(f"   Dictionary with {len(value)} items:")
                    for sub_key, sub_value in value.items():
                        # Show both type and value for dictionary contents
                        if isinstance(sub_value, (str, int, float, bool)):
                            # For simple values, show the actual value
                            print(f"     - {sub_key}: {type(sub_value).__name__} = {repr(sub_value)}")
                        elif isinstance(sub_value, (list, tuple)) and len(sub_value) <= 3:
                            # For short lists/tuples, show contents
                            print(f"     - {sub_key}: {type(sub_value).__name__} = {sub_value}")
                        elif hasattr(sub_value, 'shape') and sub_value.shape:
                            # For arrays, show shape
                            print(f"     - {sub_key}: {type(sub_value).__name__} shape{sub_value.shape}")
                        else:
                            # For complex objects, just show type
                            print(f"     - {sub_key}: {type(sub_value).__name__}")
                elif isinstance(value, (list, tuple)):
                    print(f"   Length: {len(value)}")
                    if len(value) <= 10:
                        print(f"   Content: {value}")
                    else:
                        print(f"   First 5: {value[:5]}")
                        print(f"   Last 5: {value[-5:]}")
                else:
                    print(f"   Length: {len(value)}")
            except:
                print(f"   Value: {str(value)[:100]}...")
        else:
            # Simple values
            if isinstance(value, str) and len(value) > 100:
                print(f"   Value: {value[:100]}...")
            else:
                print(f"   Value: {value}")

        # Special handling for common ML objects
        if hasattr(value, 'predict'):
            print("   📊 ML Model: Has predict method")
        if hasattr(value, 'transform'):
            print("   🔄 Transformer: Has transform method")
        if hasattr(value, 'fit'):
            print("   🎯 Estimator: Has fit method")

        # Special handling for StackingEnsemble
        if isinstance(value, StackingEnsemble):
            print("   📚 Ensemble Details:")
            print(f"     - Base models: {[type(base).__name__ for base in value.bases]}")
            print(f"     - Meta model: {type(value.meta).__name__}")
            print(f"     - Classes: {value.classes_}")
            print("   💡 StackingEnsemble: Combines multiple base models with a meta-classifier")

def edit_model_key(model_data):
    """Edit any key in the model data"""
    print("\n" + "="*60)
    print("EDIT MODEL KEY")
    print("="*60)

    # Show all keys with numbers
    print("Available keys:")
    for i, key in enumerate(model_data.keys()):
        value_type = type(model_data[key]).__name__
        print(f"{i+1}. {key} ({value_type})")

    # Get key to edit
    while True:
        key_choice = input("\nEnter key number to edit (or 'q' to quit): ").strip()
        if key_choice.lower() == 'q':
            return model_data

        try:
            key_index = int(key_choice) - 1
            if 0 <= key_index < len(model_data):
                selected_key = list(model_data.keys())[key_index]
                break
            else:
                print("Invalid key number.")
        except ValueError:
            print("Please enter a valid number.")

    current_value = model_data[selected_key]
    print(f"\nEditing key: '{selected_key}'")
    print(f"Current type: {type(current_value).__name__}")
    print(f"Current value: {current_value}")

    # Handle different types of editing
    if isinstance(current_value, dict):
        print("\nThis is a dictionary. Options:")
        print("1. Add new key-value pair")
        print("2. Edit existing key")
        print("3. Delete a key")
        print("4. Replace entire dictionary")

        dict_choice = input("Choose option (1-4): ").strip()

        if dict_choice == '1':
            new_key = input("Enter new key name: ").strip()
            new_value_str = input("Enter new value: ").strip()
            # Try to parse as different types
            try:
                # Try int
                new_value = int(new_value_str)
            except:
                try:
                    # Try float
                    new_value = float(new_value_str)
                except:
                    # Try list/tuple
                    if new_value_str.startswith('[') and new_value_str.endswith(']'):
                        new_value = eval(new_value_str)
                    else:
                        new_value = new_value_str
            current_value[new_key] = new_value
            print(f"Added {new_key}: {new_value}")

        elif dict_choice == '2':
            print("Existing keys:")
            for i, k in enumerate(current_value.keys()):
                print(f"{i+1}. {k}: {current_value[k]}")
            edit_key = input("Enter key to edit: ").strip()
            if edit_key in current_value:
                new_val = input(f"New value for {edit_key}: ").strip()
                current_value[edit_key] = new_val
                print(f"Updated {edit_key} to {new_val}")
            else:
                print("Key not found.")

        elif dict_choice == '3':
            del_key = input("Enter key to delete: ").strip()
            if del_key in current_value:
                del current_value[del_key]
                print(f"Deleted key: {del_key}")
            else:
                print("Key not found.")

        elif dict_choice == '4':
            print("⚠️  WARNING: This will replace the entire dictionary!")
            confirm = input("Are you sure? (yes/no): ").strip().lower()
            if confirm == 'yes':
                new_dict_str = input("Enter new dictionary (Python dict syntax): ").strip()
                try:
                    new_dict = eval(new_dict_str)
                    model_data[selected_key] = new_dict
                    print("Dictionary replaced.")
                    return model_data
                except:
                    print("Invalid dictionary syntax.")
            else:
                print("Operation cancelled.")

    elif isinstance(current_value, (list, tuple)):
        print("\nThis is a list/tuple. Options:")
        print("1. Append new item")
        print("2. Edit item by index")
        print("3. Delete item by index")
        print("4. Replace entire list")

        list_choice = input("Choose option (1-4): ").strip()

        if list_choice == '1':
            new_item = input("Enter new item: ").strip()
            if isinstance(current_value, list):
                current_value.append(new_item)
                print(f"Appended: {new_item}")
            else:
                print("Cannot modify tuple. Convert to list first.")

        elif list_choice == '2':
            try:
                idx = int(input("Enter index to edit: ").strip())
                if 0 <= idx < len(current_value):
                    new_val = input(f"New value for index {idx}: ").strip()
                    if isinstance(current_value, list):
                        current_value[idx] = new_val
                        print(f"Updated index {idx} to {new_val}")
                    else:
                        print("Cannot modify tuple. Convert to list first.")
                else:
                    print("Invalid index.")
            except ValueError:
                print("Invalid index.")

        elif list_choice == '3':
            try:
                idx = int(input("Enter index to delete: ").strip())
                if isinstance(current_value, list) and 0 <= idx < len(current_value):
                    deleted = current_value.pop(idx)
                    print(f"Deleted {deleted} at index {idx}")
                else:
                    print("Cannot delete from tuple or invalid index.")
            except ValueError:
                print("Invalid index.")

        elif list_choice == '4':
            print("⚠️  WARNING: This will replace the entire list/tuple!")
            confirm = input("Are you sure? (yes/no): ").strip().lower()
            if confirm == 'yes':
                new_list_str = input("Enter new list (Python list syntax): ").strip()
                try:
                    new_list = eval(new_list_str)
                    model_data[selected_key] = new_list
                    print("List replaced.")
                    return model_data
                except:
                    print("Invalid list syntax.")
            else:
                print("Operation cancelled.")

    else:
        # Simple value editing
        print("\nEnter new value:")
        new_value_str = input(f"New value for '{selected_key}': ").strip()

        # Try to preserve type
        if isinstance(current_value, int):
            try:
                new_value = int(new_value_str)
            except:
                new_value = new_value_str
        elif isinstance(current_value, float):
            try:
                new_value = float(new_value_str)
            except:
                new_value = new_value_str
        elif isinstance(current_value, bool):
            new_value = new_value_str.lower() in ['true', '1', 'yes', 'on']
        else:
            new_value = new_value_str

        model_data[selected_key] = new_value
        print(f"Updated '{selected_key}' to: {new_value}")

    return model_data

def get_user_input(prompt, current_value=None):
    """Get user input with optional current value as default"""
    if current_value is not None:
        response = input(f"{prompt} (current: {current_value}): ").strip()
        return response if response else current_value
    else:
        return input(f"{prompt}: ").strip()

def update_metadata(model_data):
    """Interactive metadata update"""
    print("\n" + "="*60)
    print("UPDATE METADATA")
    print("="*60)

    # Initialize metadata if it doesn't exist
    if 'metadata' not in model_data:
        model_data['metadata'] = {}

    metadata = model_data['metadata']

    # Standard metadata fields
    fields = [
        ('name', 'Model display name'),
        ('id', 'Model identifier'),
        ('model_name', 'Full model name'),
        ('type', 'Model type/architecture'),
        ('version', 'Model version'),
        ('description', 'Model description'),
        ('framework', 'ML framework used'),
        ('architecture', 'Detailed architecture description'),
        ('dataset', 'Training dataset name'),
        ('accuracy', 'Model accuracy'),
        ('created_date', 'Creation date'),
        ('author', 'Model author/creator')
    ]

    print("Enter new values (press Enter to keep current value):")

    for field_key, field_desc in fields:
        current = metadata.get(field_key, '')
        new_value = get_user_input(f"{field_desc}", current)
        if new_value:
            metadata[field_key] = new_value

    # Auto-update created_date if not set
    if 'created_date' not in metadata or not metadata['created_date']:
        metadata['created_date'] = datetime.now().strftime('%Y-%m-%d')

    # Auto-update last_modified
    metadata['last_modified'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("\nUpdated metadata:")
    display_metadata(model_data)

    return model_data

def save_model_data(model_data, model_path):
    """Save updated model data to joblib file"""
    try:
        # Check if this was originally a single model
        was_single_model = (len(model_data) == 1 and 'model' in model_data and 'metadata' not in model_data)

        if was_single_model and 'metadata' in model_data:
            print("⚠️  Note: Converting single model to dictionary format to include metadata.")
            print("    Original model accessible via: model_data['model']")

        # Create backup by copying the original file
        backup_path = model_path + '.backup'
        if os.path.exists(model_path):
            import shutil
            shutil.copy2(model_path, backup_path)
            print(f"Backup created: {backup_path}")

        # Save updated model
        joblib.dump(model_data, model_path)
        print(f"Model saved successfully: {model_path}")

    except Exception as e:
        print(f"Error saving model: {e}")

TEMPLATES = {
    'heart-prediction': {
        'name': 'Heart Risk Prediction',
        'id': 'heart-prediction',
        'model_name': 'Heart Risk Predictive Model',
        'method': 'AdaBoost_Classifier',
        'description': 'Heart disease risk prediction using AdaBoost with multiple weak learners',
        'version': '1.0',
        'dataset': 'Heart Disease Prediction Dataset - Kaggle',
        'modified_date': '2026-01-04',
        'author': 'Krishna Vijay G',
        'auth_url': 'https://Krishna-Vijay-G.github.io',
        'training_date': '2025-12-31',
        'performance': {
            'test_accuracy': 0.9936,
            'validation_accuracy': 0.9933,
            'roc_auc': 0.9997,
            'f1_score': 0.9936,
            'precision': 0.9936,
            'recall': 0.9936
        },
        'training_details': {
            'training_samples': 42000,
            'validation_samples': 14000,
            'test_samples': 14000,
            'total_samples': 70000,
            'features': 18,
            'classes': 2
        }
    },
    'diabetes-prediction': {
        'name': 'Diabetes Risk Prediction',
        'id': 'diabetes-prediction',
        'model_name': 'Diabetes Risk Predictive Model',
        'method': 'LightGBM_Classifier',
        'description': 'Early diabetes risk prediction using LightGBM with engineered features',
        'version': '1.0',
        'dataset': 'Early Stage Diabetes Risk Prediction - UCI',
        'modified_date': '2026-01-04',
        'author': 'Krishna Vijay G',
        'auth_url': 'https://Krishna-Vijay-G.github.io',
        'training_date': '2025-12-31',
        'performance': {
            'test_accuracy': 0.98,
            'validation_accuracy': 0.97,
            'roc_auc': 0.995,
            'f1_score': 0.98,
            'precision': 0.98,
            'recall': 0.98
        },
        'training_details': {
            'training_samples': 400,
            'validation_samples': 100,
            'test_samples': 68,
            'total_samples': 520,
            'features': 16,
            'classes': 2
        }
    },
    'skin-diagnosis': {
        'name': 'Skin Lesion Diagnosis',
        'id': 'skin-diagnosis',
        'model_name': 'Skin Lesion Diagnostic Model',
        'method': 'CNN_Voting_Ensemble',
        'description': 'Multi-class skin disease classification using CNN features with voting ensemble',
        'version': '1.0',
        'dataset': 'HAM10000 Dataset - ISIC',
        'modified_date': '2026-01-05',
        'author': 'Krishna Vijay G',
        'auth_url': 'https://Krishna-Vijay-G.github.io',
        'training_date': '2026-01-04',
        'performance': {
            'test_accuracy': 0.9884313969399179,
            'validation_accuracy': 0.8214939614311764,
            'roc_auc': 0.993,
            'f1_score': 0.9883062502087258,
            'precision': 0.9886283003228951,
            'recall': 0.9884313969399179
        },
        'training_details': {
            'training_samples': 8039,
            'validation_samples': 790,
            'test_samples': 1186,
            'total_samples': 10015,
            'features': 6224,
            'classes': 7
        }
    },
    'breast-prediction': {
        'name': 'Breast Cancer Risk Prediction',
        'id': 'breast-prediction',
        'model_name': 'BC Predictive Model',
        'method': 'XGB_Ensemble',
        'description': 'Breast cancer risk prediction using ensemble of 6 XGBoost models with calibration',
        'version': '1.0',
        'dataset': 'BCSC Prediction Factors Dataset - BCSC',
        'modified_date': '2026-01-04',
        'author': 'Krishna Vijay G',
        'auth_url': 'https://Krishna-Vijay-G.github.io',
        'training_date': '2025-12-31',
        'performance': {
            'test_accuracy': 0.813,
            'validation_accuracy': 0.821,
            'roc_auc': 0.902,
            'f1_score': 0.813,
            'precision': 0.821,
            'recall': 0.821
        },
        'training_details': {
            'training_samples': 30240,
            'validation_samples': 7560,
            'test_samples': 7560,
            'total_samples': 45360,
            'features': 10,
            'classes': 2
        }
    },
    'breast-diagnosis': {
        'name': 'Breast Cancer Tissue Diagnosis',
        'id': 'breast-diagnosis',
        'model_name': 'BC Diagnostic Model',
        'method': 'Stacking_Ensemble',
        'description': 'Breast cancer tissue diagnosis using FNA biopsy data with stacking ensemble of RF, XGB, GB, and SVM models',
        'version': '1.0',
        'dataset': 'Wisconsin Diagnosis Dataset - UCI',
        'modified_date': '2026-01-04',
        'author': 'Krishna Vijay G',
        'auth_url': 'https://Krishna-Vijay-G.github.io',
        'training_date': '2025-12-31',
        'performance': {
            'test_accuracy': 0.972,
            'validation_accuracy': 0.974,
            'roc_auc': 0.994,
            'f1_score': 0.972,
            'precision': 0.976,
            'recall': 0.976
        },
        'training_details': {
            'training_samples': 455,
            'validation_samples': 114,
            'test_samples': 114,
            'total_samples': 569,
            'features': 5,
            'classes': 2
        }
    }
}


def apply_template_to_file(template: dict, model_path: str) -> bool:
    """Apply template metadata to a model joblib file"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False
    try:
        data = joblib.load(model_path)
        if not isinstance(data, dict):
            # wrap single model object
            data = {'model': data}
        # Merge metadata, prefer existing fields if present
        if 'metadata' not in data or not isinstance(data['metadata'], dict):
            data['metadata'] = {}
        data['metadata'].update(template)
        # Backup
        import shutil
        shutil.copy2(model_path, model_path + '.metadata_backup')
        joblib.dump(data, model_path)
        print(f"Updated metadata for {model_path}")
        return True
    except Exception as e:
        print(f"Failed to update {model_path}: {e}")
        return False


def apply_templates():
    """Apply all available templates to known model files in the repo"""
    base = os.path.join(os.path.dirname(__file__), '..', 'models')
    mapping = {
        'heart-prediction': os.path.join(base, 'Heart Risk Predictive Model', 'heart-prediction.joblib'),
        'diabetes-prediction': os.path.join(base, 'Diabetes Risk Predictive Model', 'diabetes-prediction.joblib'),
        'skin-diagnosis': os.path.join(base, 'Skin Lesion Diagnostic Model', 'skin-diagnosis.joblib'),
        'breast-prediction': os.path.join(base, 'BC Predictive Model', 'breast-prediction.joblib'),
        'breast-diagnosis': os.path.join(base, 'BC Diagnostic Model', 'breast-diagnosis.joblib')
    }

    for key, path in mapping.items():
        tpl = TEMPLATES.get(key)
        if tpl and os.path.exists(path):
            print(f"Applying template '{key}' to {path}")
            apply_template_to_file(tpl, path)
        else:
            print(f"Skipping {key}; file not found or template missing: {path}")


def main():
    """Main function"""
    print("Model Metadata Manager")
    print("======================")

    # Get model path
    model_path = input("Enter model file path (e.g., backend/models/Dermatology_Model/derm_model.joblib) or leave blank to use templates: ").strip()

    if not model_path:
        choice = input("Apply predefined templates to known models? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            apply_templates()
        else:
            print("No path provided and templates not applied. Exiting.")
        return

    # Load model
    model_data = load_model_data(model_path)
    if model_data is None:
        return

    # Display current metadata
    display_metadata(model_data)

    # Ask if user wants to inspect all keys
    inspect_choice = input("\nDo you want to inspect all keys in detail? (y/n): ").strip().lower()

    if inspect_choice in ['y', 'yes']:
        inspect_model_keys(model_data)

    # Ask if user wants to edit any key
    edit_choice = input("\nDo you want to edit any key in the model? (y/n): ").strip().lower()
    changes_made = False

    if edit_choice in ['y', 'yes']:
        model_data = edit_model_key(model_data)
        changes_made = True

    # Ask if user wants to update metadata
    update_choice = input("\nDo you want to update the metadata? (y/n): ").strip().lower()

    if update_choice in ['y', 'yes']:
        # Update metadata
        model_data = update_metadata(model_data)
        changes_made = True

    # Save changes if any were made
    if changes_made:
        save_choice = input("\nSave all changes? (y/n): ").strip().lower()
        if save_choice in ['y', 'yes']:
            save_model_data(model_data, model_path)
            print("All changes saved successfully!")
        else:
            print("Changes not saved.")
    else:
        print("No changes made.")

    print("\nDone!")

if __name__ == "__main__":
    import sys
    if '--apply-templates' in sys.argv:
        apply_templates()
    else:
        main()