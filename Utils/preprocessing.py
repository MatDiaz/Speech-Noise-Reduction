import os
import tarfile
import shutil
import urllib.request
from pathlib import Path

def download_and_extract(url, dir):
  os.makedirs(dir, exist_ok=True)
  filename = os.path.basename(url)
  tar_path = os.path.join(dir, filename)

  print(f"Descargando desde {url}")
  urllib.request.urlretrieve(url, tar_path)
  print(f"Completado {url}")

  print(f"Desempaquetando")
  with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=dir)
  print(f"Completado")

  print(f"Limpiando")
  target_path = Path(os.path.join(dir, Path(filename).with_suffix("").with_suffix("")))
  target_path.mkdir(exist_ok=True)

  for each_file in Path(dir).rglob('*.flac'):
      shutil.move(str(each_file), str(target_path / each_file.name))

  # Borrar carpetas y archivos innecesarios
  shutil.rmtree(dir + "/LibriSpeech")
  tar_path = Path(tar_path)

  if tar_path.exists():
      tar_path.unlink(missing_ok=True)