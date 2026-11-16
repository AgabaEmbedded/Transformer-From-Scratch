from pathlib import Path
def get_config():
  return {
      "batch_size": 16,
      "num_epochs": 1,
      "lr": 10**-4,
      "seq_len": 350,
      "d_model": 512,
      "model_folder": "weights",
      "model_basename": "tmodel_",
      "preload": True,
      "tokenizer_file": "tokenizer_.json",
      "experiment_name": "runs/tmodel"
  }

def get_weights_file_path(config, epoch: str):
  model_folder = Path(config['model_folder'])
  model_filename = Path(config["model_basename"])
  model_folder.mkdir(exist_ok = True)
  return str(Path(".") / model_folder / model_filename)