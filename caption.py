"""
Copyright [2022-2023] Victor C Hall

Licensed under the GNU Affero General Public License;
You may not use this code except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/agpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

from PIL import Image
import argparse
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, GitProcessor, GitForCausalLM, AutoModel, AutoProcessor

import torch
from pynvml import *

import time
from colorama import Fore, Style

SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]

def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024 / 1024

def create_blip2_processor(model_name, device, dtype=torch.float16):
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    print(f"BLIP2 Model loaded: {model_name}")
    return processor, model

def create_git_processor(model_name, device, dtype=torch.float16):
    processor = GitProcessor.from_pretrained(model_name)
    model = GitForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    print(f"GIT Model loaded: {model_name}")
    return processor, model

def create_auto_processor(model_name, device, dtype=torch.float16):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    print("Auto Model loaded")
    return processor, model

def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    dtype = torch.float32 if args.force_cpu else torch.float16

    if "salesforce/blip2-" in args.model.lower():
        print(f"Using BLIP2 model: {args.model}")
        processor, model = create_blip2_processor(args.model, device, dtype)
    elif "microsoft/git-" in args.model.lower():
        print(f"Using GIT model: {args.model}")
        processor, model = create_git_processor(args.model, device, dtype)
    else:
        # try to use auto model?  doesn't work with blip/git
        processor, model = create_auto_processor(args.model, device, dtype)

    print(f"GPU memory used, after loading model: {get_gpu_memory_map()} MB")

    # os.walk all files in args.data_root recursively
    for root, dirs, files in os.walk(args.data_root):
        for file in files:
                        # get file extension
            ext = os.path.splitext(file)[1]
            if ext.lower() in SUPPORTED_EXT:
                full_file_path = os.path.join(root, file)
                try:
                    image = Image.open(full_file_path)
                    start_time = time.time()

                    inputs = processor(images=image, return_tensors="pt", max_new_tokens=args.max_new_tokens).to(device, dtype)

                    generated_ids = model.generate(**inputs)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    print(f"file: {file}, caption: {generated_text}")
                    exec_time = time.time() - start_time
                    print(f"  Time for last caption: {exec_time} sec.  GPU memory used: {get_gpu_memory_map()} MB")

                    # get bare name
                    name = os.path.splitext(full_file_path)[0]
                    if not os.path.exists(name):
                        with open(f"{name}.txt", "w") as f:
                            f.write(generated_text)
                except (IOError, OSError):
                    print(f"Skipping unidentified image: {file}")
                    continue

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Image Captioning")
parser.add_argument("--data_root", type=str, default="images", help="Root directory of the image data")
parser.add_argument("--model", type=str, default="salesforce/blip2-en-base", help="Model name or path")
parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens in the generated caption")
args = parser.parse_args()

# Run the main function
main(args)
