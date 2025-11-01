"""Model configuration and management utilities"""
import os
import shutil
import platform
import logging
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox, QInputDialog

class ModelConfig:
    """Model configuration and mapping utilities"""
    
    # Known model mappings for variants
    KNOWN_MODELS = {
        'mlx-community/whisper-large-v3-turbo': 'openai/whisper-large-v3-turbo',
        'mlx-community/whisper-large-v3-turbo-q4': 'openai/whisper-large-v3-turbo',
        'mlx-community/whisper-tiny-mlx-4bit': 'openai/whisper-tiny',
        'mlx-community/whisper-base-mlx-4bit': 'openai/whisper-base',
        'mlx-community/whisper-small-mlx-q4': 'openai/whisper-small',
        'mlx-community/whisper-medium-mlx-4bit': 'openai/whisper-medium',
        'mlx-community/whisper-large-v3-mlx-4bit': 'openai/whisper-large-v3',
        'mlx-community/whisper-large-v3-mlx': 'openai/whisper-large-v3',
        'cstr/whisper-large-v3-turbo-int8_float32': 'openai/whisper-large-v3-turbo',
        'SYSTRAN/faster-whisper-large-v1': 'openai/whisper-large-v2',
        'GalaktischeGurke/primeline-whisper-large-v3-german-ct2': 'openai/whisper-large-v3'
    }
    
    @classmethod
    def get_original_model_id(cls, model_id):
        """Get the original OpenAI model ID from a variant"""
        # Check known models first
        if model_id in cls.KNOWN_MODELS:
            return cls.KNOWN_MODELS[model_id]
        
        model_id_lower = model_id.lower()
        
        # Handle special cases
        if model_id.startswith("openai/whisper-"):
            return model_id
        
        if "endpoint" in model_id_lower:
            return "openai/whisper-large-v2"
        
        # Determine version
        if "v3_turbo" in model_id_lower or "v3-turbo" in model_id_lower:
            base = "openai/whisper-large-v3-turbo"
        elif "v3" in model_id_lower:
            base = "openai/whisper-large-v3"
        elif "v2" in model_id_lower:
            base = "openai/whisper-large-v2"
        elif "v1" in model_id_lower:
            base = "openai/whisper-large-v1"
        else:
            base = "openai/whisper"
        
        # Determine size if base is generic
        if base == "openai/whisper":
            if "large" in model_id_lower:
                size = "large"
            elif "medium" in model_id_lower:
                size = "medium"
            elif "small" in model_id_lower:
                size = "small"
            elif "base" in model_id_lower:
                size = "base"
            elif "tiny" in model_id_lower:
                size = "tiny"
            else:
                size = "large"
            base = f"{base}-{size}"
        
        # Check for language specificity
        if "_en" in model_id_lower or ".en" in model_id_lower:
            lang = ".en"
        else:
            lang = ""
        
        return f"{base}{lang}"
    
    @classmethod
    def find_whisper_cpp_executable(cls):
        """Find the whisper.cpp executable with fallbacks"""
        system = platform.system()
        executable_names = ["whisper", "whisper.exe"] if system == "Windows" else ["whisper"]
        
        # Add additional names
        if system == "Windows":
            executable_names.extend(["main.exe", "whisper-cli.exe"])
        else:
            executable_names.extend(["main", "whisper-cli"])
        
        # Define search paths
        search_paths = [
            os.getcwd(),
            os.path.join(os.getcwd(), "whisper.cpp"),
            os.path.join(os.getcwd(), "bin"),
            os.path.join(os.getcwd(), "build"),
            os.path.expanduser("~"),
            os.path.join(os.path.expanduser("~"), "whisper.cpp"),
            os.path.join(os.path.expanduser("~"), "bin"),
        ]
        
        # Platform-specific paths
        if system == "Windows":
            search_paths.extend([
                os.path.join(os.getcwd(), "build", "Release"),
                os.path.join(os.getcwd(), "build", "Debug"),
                "C:\\Program Files\\whisper.cpp",
                "C:\\Program Files (x86)\\whisper.cpp",
            ])
        elif system in ["Darwin", "Linux"]:
            search_paths.extend([
                "/usr/local/bin",
                "/usr/bin",
                "/opt/whisper.cpp/bin",
            ])
        
        # Search for executable
        for path in search_paths:
            if os.path.exists(path):
                for exe_name in executable_names:
                    exe_path = os.path.join(path, exe_name)
                    if os.path.isfile(exe_path) and os.access(exe_path, os.X_OK):
                        logging.info(f"Found whisper.cpp executable at: {exe_path}")
                        return exe_path
        
        logging.error("No whisper.cpp executable found in search paths")
        return None
    
    @classmethod
    def find_or_download_whisper_cpp_model(cls, model_id):
        """Find or download a whisper.cpp model"""
        model_file = model_id
        if not model_file.startswith('ggml-'):
            model_file = f'ggml-{model_file}'
        if not model_file.endswith('.bin'):
            model_file = f'{model_file}.bin'
        
        # Search paths
        search_paths = [
            os.getcwd(),
            os.path.join(os.getcwd(), 'models'),
            os.path.expanduser('~'),
            os.path.join(os.path.expanduser('~'), 'models'),
            os.path.join(os.path.expanduser('~'), 'whisper.cpp', 'models'),
        ]
        
        # Look for existing model
        for path in search_paths:
            if os.path.exists(path):
                model_path = os.path.join(path, model_file)
                if os.path.exists(model_path):
                    logging.info(f"Model file '{model_file}' found at: {model_path}")
                    return model_path
        
        # Download if not found
        models_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, model_file)
        
        try:
            import requests
            base_url = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/'
            url = base_url + model_file
            logging.info(f'Downloading {model_file} from {url}')
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f'Model downloaded to {model_path}')
            return model_path
        except Exception as e:
            logging.error(f"Failed to download model: {str(e)}")
            if os.path.exists(model_path) and os.path.getsize(model_path) == 0:
                os.remove(model_path)
            raise


class CTranslate2ModelConverter:
    """Utilities for converting models to CTranslate2 format"""
    
    @staticmethod
    def find_or_convert_model(model_id, parent_widget=None):
        """Find or convert a CTranslate2 model
        
        Args:
            model_id: Model identifier
            parent_widget: Parent widget for dialogs (optional)
            
        Returns:
            Tuple of (model_dir, original_model_id) or (None, None) on error
        """
        # Extract the original model ID
        original_model_id = ModelConfig.get_original_model_id(model_id)
        logging.info(f"Original model ID determined as: {original_model_id}")

        model_dir_name = model_id.replace('/', '_')
        local_model_dir = os.path.join(os.getcwd(), 'ctranslate2_models', model_dir_name)
        local_model_bin_path = os.path.join(local_model_dir, 'model.bin')

        # First, check if model.bin exists locally and is not empty
        if os.path.exists(local_model_bin_path) and os.path.getsize(local_model_bin_path) > 0:
            if os.path.islink(local_model_bin_path):
                # If it's a symlink, replace it with the actual file
                real_path = os.path.realpath(local_model_bin_path)
                os.remove(local_model_bin_path)
                shutil.copy(real_path, local_model_bin_path)
                logging.info(f"Replaced symlink with actual file for model.bin in: {local_model_dir}")
            else:
                logging.info(f"Model already converted and exists locally in: {local_model_dir}")
            return local_model_dir, original_model_id
        else:
            # If the directory exists but model.bin is missing or empty, remove the directory
            if os.path.exists(local_model_dir):
                shutil.rmtree(local_model_dir)
            os.makedirs(local_model_dir, exist_ok=True)

        # Check if the model is already in CTranslate2 format in the Hugging Face repo
        try:
            try:
                from huggingface_hub import hf_hub_download, HfApi
            except ImportError:
                QMessageBox.critical(
                    parent_widget, 
                    "Error", 
                    "huggingface_hub package is not installed. Cannot proceed."
                )
                logging.error("huggingface_hub package is not installed.")
                return None, None

            api = HfApi()
            model_files = api.list_repo_files(model_id)
            if 'model.bin' in model_files:
                logging.info(f"Found pre-converted CTranslate2 model in Hugging Face repo: {model_id}")
                # Download the model.bin and other necessary files
                for file in ['model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json', 'preprocessor_config.json']:
                    if file in model_files:
                        file_path = hf_hub_download(repo_id=model_id, filename=file, local_dir=local_model_dir)
                        # If the downloaded file is a symlink, replace it with the actual file
                        if os.path.islink(file_path):
                            real_path = os.path.realpath(file_path)
                            os.remove(file_path)
                            shutil.copy(real_path, file_path)
                return local_model_dir, original_model_id

        except Exception as e:
            logging.error(f"Error while checking Hugging Face repo: {e}")
            logging.info("Proceeding with local model conversion...")

        # If we reach here, we need to convert the model
        logging.info(f"Model not found in CTranslate2 format or couldn't be downloaded. Converting {model_id} to CTranslate2 format...")

        # Ask for quantization
        quantization, ok_pressed = QInputDialog.getItem(
            parent_widget,
            "Select Quantization",
            "Choose quantization for model conversion:",
            ["float32", "int8_float16", "int16", "int8"],
            1,
            False
        )
        if not ok_pressed:
            QMessageBox.information(
                parent_widget, 
                "Information", 
                "Quantization selection cancelled by user."
            )
            logging.info("Quantization selection cancelled by user.")
            return None, None

        reply = QMessageBox.question(
            parent_widget,
            "Model Conversion Required",
            f"The model {model_id} needs to be converted to CTranslate2 format with quantization '{quantization}'. This may take several minutes. Do you want to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            QMessageBox.information(
                parent_widget, 
                "Information", 
                "Model conversion cancelled by user."
            )
            logging.info("Model conversion cancelled by user.")
            return None, None

        # Perform the model conversion
        try:
            CTranslate2ModelConverter.convert_model(model_id, local_model_dir, quantization, parent_widget)
            logging.info(f"Model converted and saved to: {local_model_dir}")
        except Exception as e:
            logging.error(f"Error during model conversion: {e}")
            QMessageBox.critical(
                parent_widget, 
                "Error", 
                f"Error during model conversion: {str(e)}"
            )
            return None, None

        # Download the preprocessor files and save them in the same directory
        try:
            try:
                from transformers import WhisperProcessor
            except ImportError:
                QMessageBox.critical(
                    parent_widget, 
                    "Error", 
                    "transformers package is not installed. Cannot load the preprocessor files."
                )
                logging.error("transformers package is not installed.")
                return None, None

            logging.info(f"Downloading preprocessor files for original model: {original_model_id}")
            preprocessor = WhisperProcessor.from_pretrained(original_model_id)
            preprocessor.save_pretrained(local_model_dir)
            logging.info(f"Preprocessor files saved to: {local_model_dir}")
        except Exception as e:
            logging.error(f"Error downloading preprocessor files: {e}")
            QMessageBox.critical(
                parent_widget, 
                "Error", 
                f"Error downloading preprocessor files: {str(e)}"
            )
            return None, None

        return local_model_dir, original_model_id
    
    @staticmethod
    def convert_model(model_id, output_dir, quantization, parent_widget=None):
        """Convert model to CTranslate2 format
        
        Args:
            model_id: Model identifier
            output_dir: Output directory for converted model
            quantization: Quantization type
            parent_widget: Parent widget for dialogs (optional)
        """
        try:
            from ctranslate2.converters.transformers import TransformersConverter
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError:
            QMessageBox.critical(
                parent_widget, 
                "Error", 
                "Required packages for model conversion are not installed. Please install 'ctranslate2' and 'transformers'."
            )
            logging.error("Required packages 'ctranslate2' and 'transformers' are not installed.")
            return

        logging.info(f"Loading model {model_id} for conversion...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)

        logging.info(f"Converting model to CTranslate2 format...")
        converter = TransformersConverter(model, processor)
        converter.convert(output_dir, quantization=quantization, force=True)

        # Verify that the model.bin file was created and is not a symlink
        model_bin_path = os.path.join(output_dir, 'model.bin')
        if not os.path.exists(model_bin_path) or os.path.getsize(model_bin_path) == 0:
            error_msg = f"Failed to convert model. model.bin not found or empty in {output_dir}"
            logging.error(error_msg)
            QMessageBox.critical(parent_widget, "Error", error_msg)
            return
        if os.path.islink(model_bin_path):
            real_path = os.path.realpath(model_bin_path)
            os.remove(model_bin_path)
            shutil.copy(real_path, model_bin_path)

        logging.info(f"Model successfully converted and saved to: {output_dir}")


# Convenience functions
def get_original_model_id(model_id):
    """Get original model ID - convenience function"""
    return ModelConfig.get_original_model_id(model_id)

def find_whisper_cpp_executable():
    """Find whisper.cpp executable - convenience function"""
    return ModelConfig.find_whisper_cpp_executable()

def find_or_download_whisper_cpp_model(model_id):
    """Find or download whisper.cpp model - convenience function"""
    return ModelConfig.find_or_download_whisper_cpp_model(model_id)