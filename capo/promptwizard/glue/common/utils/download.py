import os
import requests

from pathlib import Path
from urllib.parse import urlparse
from glue.common.constants.str_literals import DirNames
from glue.common.utils.logging import get_glue_logger

logger = get_glue_logger(__name__)

def download_model(url):
    cwd = os.getcwd()
    dirs = Path(cwd).parts
    idx = 0
    if DirNames.PACKAGE_BASE_DIR in dirs:
        idx = dir.index(DirNames.PACKAGE_BASE_DIR)
    download_path = os.path.join(*dir[:idx+1], DirNames.MODEL_DIR)
    os.makedirs(download_path, exist_ok=True)

    parsed_url = urlparse(url)
    model_filename = os.path.basename(parsed_url.path)

    model_path = os.path.join(download_path, model_filename)
    if not os.path.exists(model_path):
        r = requests.get(url, stream=True)
        if r.ok:
            with os.open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 8):
                    if chunk:
                        f.write(chunk)
                        f.flush()
    
    return model_path


