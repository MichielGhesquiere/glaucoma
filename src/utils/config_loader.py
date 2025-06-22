import yaml
import os
import logging

# Configure basic logging for the loader itself
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str) -> dict:
    """
    Loads configuration settings from a YAML file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
        Exception: For other potential errors during file reading.
    """
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at: {config_path}")
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
        logging.info(f"Successfully loaded configuration from: {config_path}")
        return config if config is not None else {} # Return empty dict if file is empty
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error reading configuration file {config_path}: {e}")
        raise

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Create a dummy config file for testing
    dummy_config = {
        'data': {
            'raw_path': '/path/to/raw',
            'processed_path': '/path/to/processed'
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'epochs': 50
        },
        'model': {
            'name': 'resnet50',
            'num_classes': 2
        }
    }
    dummy_path = 'dummy_config.yaml'
    try:
        with open(dummy_path, 'w') as f:
            yaml.dump(dummy_config, f)

        # Test loading
        loaded_config = load_config(dummy_path)
        print("Loaded config:")
        print(loaded_config)
        print("\nAccessing a value:", loaded_config.get('training', {}).get('learning_rate'))

    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        # Clean up dummy file
        if os.path.exists(dummy_path):
            os.remove(dummy_path)