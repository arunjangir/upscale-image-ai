# -*- coding: utf-8 -*-

#!/usr/bin/env python3
"""
High-Performance Real-ESRGAN Gradio API Server
Optimized for 16GB GPU RAM systems
"""
import os
import sys

# CRITICAL: Set threading environment variables BEFORE any imports
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

def set_torch_threads():
    """Set PyTorch threading before any operations"""
    import torch
    torch.set_num_threads(8)
    if hasattr(torch, 'set_num_interop_threads'):
        torch.set_num_interop_threads(4)


# Now safe to import torch and other modules
import torch
set_torch_threads()

import time
import json
import shutil
import subprocess
import threading
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gradio as gr
import psutil
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import logging
from datetime import datetime
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

import threading

# Load environment variables
load_dotenv()

# Configuration for 16GB GPU system
IDLE_SHUTDOWN_SECONDS = int(os.getenv('IDLE_SHUTDOWN_SECONDS', '1800'))
UPLOAD_FOLDER = 'temp_upload'
RESULT_FOLDER = 'temp_results'
CPU_GPU_THRESHOLD = 85
API_KEY = os.getenv('API_KEY', '12345')
WEBHOOK_URL = os.getenv('WEBHOOK_URL', '')
PROCESSING_TIMEOUT = 900

# Optimized settings for 16GB GPU
MAX_TILE_SIZE = int(os.getenv('MAX_TILE_SIZE', '1024'))  # Large tiles for 16GB GPU
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '4'))  # Process 4 images at once
NUM_WORKERS = 8  # Fixed for optimal performance
PRELOAD_MODELS = True
USE_HALF_PRECISION = False  # Use FP16 for speed
GPU_MEMORY_FRACTION = 0.95  # Use 95% of 16GB
AGGRESSIVE_CLEANUP = True

# Cloudflare R2 Configuration
R2_ACCESS_KEY = os.getenv('R2_ACCESS_KEY')
R2_SECRET_KEY = os.getenv('R2_SECRET_KEY')
R2_ENDPOINT = os.getenv('R2_ENDPOINT')
R2_BUCKET = os.getenv('R2_BUCKET')


#lambda webhook
LAMBDA_WEBHOOK_URL = os.getenv('LAMBDA_WEBHOOK_URL', '')  # Your Lambda function URL


# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realesrgan_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global variables
last_activity_time = time.time()
shutdown_timer = None
model_cache = {}

class GPUOptimizer:
    """GPU optimization for 16GB VRAM systems"""
    
    @staticmethod
    def optimize_cuda():
        """Apply CUDA optimizations for 16GB GPU"""
        if torch.cuda.is_available():
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory fraction for 16GB GPU
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
            
            # Enable memory mapping for large models
            torch.backends.cuda.enable_flash_sdp(True)
            
            # Clear cache
            torch.cuda.empty_cache()
            
            logger.info(f"CUDA optimized for 16GB GPU - using {GPU_MEMORY_FRACTION*100}% VRAM")
            return True
        return False
    
    @staticmethod
    def get_gpu_memory_info():
        """Get detailed GPU memory info"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            
            return {
                'allocated_gb': round(allocated, 2),
                'reserved_gb': round(reserved, 2),
                'total_gb': round(total, 2),
                'free_gb': round(free, 2),
                'usage_percent': round((reserved / total) * 100, 1)
            }
        return None
    
    @staticmethod
    def cleanup_gpu_memory():
        """Aggressive GPU memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

class HighPerformanceR2Handler:
    """Ultra-fast R2 operations for large files"""
    
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            region_name='auto',
            config=boto3.session.Config(
                max_pool_connections=50,
                retries={'max_attempts': 3},
                connect_timeout=10,
                read_timeout=60
            )
        )
        logger.info("R2Handler initialized with high-performance config")
    
    def download_files_parallel(self, file_list: List[Tuple[str, str]]) -> List[bool]:
        """Ultra-fast parallel downloads"""
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_file = {
                executor.submit(self._download_single, remote_path, local_path): (remote_path, local_path)
                for remote_path, local_path in file_list
            }
            
            for future in as_completed(future_to_file):
                remote_path, local_path = future_to_file[future]
                try:
                    success = future.result()
                    results.append(success)
                    if success:
                        file_size = os.path.getsize(local_path) / 1024**2  # MB
                        logger.info(f"Downloaded: {os.path.basename(remote_path)} ({file_size:.1f}MB)")
                except Exception as e:
                    logger.error(f"Download failed {remote_path}: {e}")
                    results.append(False)
        
        total_time = time.time() - start_time
        total_mb = sum(os.path.getsize(local) / 1024**2 for _, local in file_list if os.path.exists(local))
        speed = total_mb / total_time if total_time > 0 else 0
        logger.info(f"Download complete: {len(file_list)} files, {total_mb:.1f}MB in {total_time:.2f}s ({speed:.1f}MB/s)")
        return results
    
    def _download_single(self, remote_path: str, local_path: str) -> bool:
        """Download single file with error handling"""
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(R2_BUCKET, remote_path, local_path)
            return os.path.exists(local_path)
        except Exception as e:
            logger.error(f"Download error for {remote_path}: {e}")
            return False
    
    def upload_files_parallel(self, file_list: List[Tuple[str, str]]) -> List[bool]:
        """Ultra-fast parallel uploads"""
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_file = {
                executor.submit(self._upload_single, local_path, remote_path): (local_path, remote_path)
                for local_path, remote_path in file_list
            }
            
            for future in as_completed(future_to_file):
                local_path, remote_path = future_to_file[future]
                try:
                    success = future.result()
                    results.append(success)
                    if success:
                        file_size = os.path.getsize(local_path) / 1024**2  # MB
                        logger.info(f"Uploaded: {os.path.basename(remote_path)} ({file_size:.1f}MB)")
                except Exception as e:
                    logger.error(f"Upload failed {remote_path}: {e}")
                    results.append(False)
        
        total_time = time.time() - start_time
        total_mb = sum(os.path.getsize(local) / 1024**2 for local, _ in file_list if os.path.exists(local))
        speed = total_mb / total_time if total_time > 0 else 0
        logger.info(f"Upload complete: {len(file_list)} files, {total_mb:.1f}MB in {total_time:.2f}s ({speed:.1f}MB/s)")
        return results
    
    def _upload_single(self, local_path: str, remote_path: str) -> bool:
        """Upload single file with compression check"""
        try:
            if not os.path.exists(local_path):
                return False
            self.s3_client.upload_file(local_path, R2_BUCKET, remote_path)
            return True
        except Exception as e:
            logger.error(f"Upload error for {remote_path}: {e}")
            return False

class SystemMonitor:
    """Advanced system monitoring for high-performance systems"""
    
    @staticmethod
    def get_cpu_usage() -> float:
        return psutil.cpu_percent(interval=0.1)
    
    @staticmethod
    def get_gpu_usage() -> float:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0.0
    
    @staticmethod
    def get_available_ram_gb() -> float:
        return psutil.virtual_memory().available / (1024**3)
    
    @classmethod
    def should_queue(cls) -> bool:
        cpu_usage = cls.get_cpu_usage()
        gpu_usage = cls.get_gpu_usage()
        return cpu_usage > CPU_GPU_THRESHOLD or gpu_usage > CPU_GPU_THRESHOLD
    
    @classmethod
    def log_performance_status(cls):
        """Log comprehensive performance metrics"""
        ram_gb = cls.get_available_ram_gb()
        cpu = cls.get_cpu_usage()
        gpu = cls.get_gpu_usage()
        gpu_mem = GPUOptimizer.get_gpu_memory_info()
        
        if gpu_mem:
            logger.info(f"System Status | CPU: {cpu}% | RAM: {ram_gb:.1f}GB | "
                       f"GPU: {gpu}% | VRAM: {gpu_mem['usage_percent']}% ({gpu_mem['free_gb']:.1f}GB free)")
        else:
            logger.info(f"System Status | CPU: {cpu}% | RAM: {ram_gb:.1f}GB")

def run_ultra_fast_realesrgan(model_name: str, scale: float = 4.0, face_enhance: bool = True) -> bool:
    """Ultra-optimized Real-ESRGAN execution for 16GB GPU"""
    try:
        logger.info(f"ULTRA-FAST Real-ESRGAN | Model: {model_name} | Scale: {scale}x")
        SystemMonitor.log_performance_status()
        
        # GPU memory optimization
        GPUOptimizer.cleanup_gpu_memory()
        
        cmd = [
            'python3', 'inference_realesrgan.py',
            '-n', model_name,
            '-i', UPLOAD_FOLDER,
            '--outscale', str(scale),
            '-o', RESULT_FOLDER,
            '--suffix', 'out'
        ]
        
        # Optimizations for 16GB GPU
        if MAX_TILE_SIZE > 512:
            cmd.extend(['--tile', str(MAX_TILE_SIZE)])
            logger.info(f"Using large tiles: {MAX_TILE_SIZE}px for 16GB GPU")
        else:
            logger.info("Using NO TILING for maximum speed with 16GB GPU")
        
        if face_enhance:
            cmd.append('--face_enhance')
        
        # GPU optimization
        if torch.cuda.is_available():
            cmd.extend(['--gpu-id', '0'])
            # Add half precision if supported
            if USE_HALF_PRECISION:
                cmd.append('--fp16')
        
        # Performance environment
        env = os.environ.copy()
        env.update({
            'CUDA_LAUNCH_BLOCKING': '0',
            'CUDA_CACHE_DISABLE': '0',
            'CUDA_VISIBLE_DEVICES': '0',
            'OMP_NUM_THREADS': str(NUM_WORKERS),
            'MKL_NUM_THREADS': str(NUM_WORKERS)
        })
        
        logger.info(f"Executing: {' '.join(cmd)}")
        start_time = time.time()
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=PROCESSING_TIMEOUT,
            env=env
        )
        
        process_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"ULTRA-FAST processing completed in {process_time:.2f}s")
            if AGGRESSIVE_CLEANUP:
                GPUOptimizer.cleanup_gpu_memory()
            return True
        else:
            logger.error(f"Processing failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Ultra-fast processing error: {str(e)}", exc_info=True)
        if AGGRESSIVE_CLEANUP:
            GPUOptimizer.cleanup_gpu_memory()
        return False

def batch_process_large_images(files: List[str], model_name: str, scale: float, face_enhance: bool) -> bool:
    """Process large images in optimized batches for 16GB GPU"""
    batches = [files[i:i+BATCH_SIZE] for i in range(0, len(files), BATCH_SIZE)]
    
    for batch_idx, batch in enumerate(batches, 1):
        logger.info(f"Processing batch {batch_idx}/{len(batches)} with {len(batch)} images")
        
        # Create batch directory
        batch_dir = f"{UPLOAD_FOLDER}_batch_{batch_idx}"
        os.makedirs(batch_dir, exist_ok=True)
        
        # Move files to batch
        for file_path in batch:
            filename = os.path.basename(file_path)
            shutil.move(file_path, os.path.join(batch_dir, filename))
        
        # Process batch
        original_upload = UPLOAD_FOLDER
        UPLOAD_FOLDER = batch_dir
        
        success = run_ultra_fast_realesrgan(model_name, scale, face_enhance)
        
        UPLOAD_FOLDER = original_upload
        
        if not success:
            return False
        
        # Move results
        for result_file in os.listdir(RESULT_FOLDER):
            if f'batch_{batch_idx}' not in result_file:
                continue
            src = os.path.join(RESULT_FOLDER, result_file)
            dst = os.path.join(RESULT_FOLDER, result_file.replace(f'_batch_{batch_idx}', ''))
            shutil.move(src, dst)
        
        # Cleanup
        shutil.rmtree(batch_dir, ignore_errors=True)
        
        if AGGRESSIVE_CLEANUP:
            GPUOptimizer.cleanup_gpu_memory()
    
    return True

# Utility functions
def update_activity():
    global last_activity_time
    last_activity_time = time.time()

def reset_shutdown_timer():
    global shutdown_timer
    if shutdown_timer:
        shutdown_timer.cancel()
    shutdown_timer = threading.Timer(IDLE_SHUTDOWN_SECONDS, shutdown_system)
    shutdown_timer.start()

def shutdown_system():
    current_time = time.time()
    idle_time = current_time - last_activity_time
    
    if idle_time >= IDLE_SHUTDOWN_SECONDS:
        logger.warning(f"Shutting down after {idle_time:.0f}s of inactivity")
        os.system("sudo shutdown -h now")
    else:
        reset_shutdown_timer()

def setup_folders():
    """Setup and clean working directories"""
    for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    logger.info("Working directories initialized")

def validate_api_key(api_key: str) -> bool:
    return api_key == API_KEY

def send_webhook(data: dict):
    """Send completion webhook to Lambda"""
    if LAMBDA_WEBHOOK_URL:
        try:
            # Send to Lambda function
            webhook_data = {
                'status': 'completed',
                'processed_files': data.get('processed_files', []),
                'total_files': data.get('total_files', 0),
                'timestamp': datetime.now().isoformat()
            }
            response = requests.post(LAMBDA_WEBHOOK_URL, json=webhook_data, timeout=10)
            logger.info(f"Lambda webhook sent: {response.status_code}")
        except Exception as e:
            logger.error(f"Lambda webhook failed: {e}")
    
    # Keep existing webhook functionality
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json=data, timeout=10)
        except Exception as e:
            logger.error(f"Original webhook failed: {e}")

def process_images(
    api_key: str,
    files_data: str,
    model_name: str = "RealESRGAN_x4plus",
    scale: float = 4.0,
    face_enhance: bool = True,
    use_ultra_mode: bool = True
) -> dict:
    """Main processing function optimized for 16GB GPU"""
    
    update_activity()
    reset_shutdown_timer()
    SystemMonitor.log_performance_status()
    
    if not validate_api_key(api_key):
        return {"error": "Invalid API key", "status": "failed"}

    if SystemMonitor.should_queue():
        return {"error": "System busy", "status": "queued"}

    try:
        files_info = json.loads(files_data)
        logger.info(f"Processing {len(files_info)} files in ULTRA mode: {use_ultra_mode}")
        
        # Initialize GPU optimizations
        if use_ultra_mode:
            GPUOptimizer.optimize_cuda()
        
        setup_folders()
        r2_handler = HighPerformanceR2Handler()
        
        # Download files
        download_list = [
            (file_info['input_file'], os.path.join(UPLOAD_FOLDER, os.path.basename(file_info['input_file'])))
            for file_info in files_info
        ]
        
        download_results = r2_handler.download_files_parallel(download_list)
        
        if not all(download_results):
            return {"error": "Download failed", "status": "failed"}
        
        # Process images
        if use_ultra_mode and len(files_info) > BATCH_SIZE:
            input_files = [dl[1] for dl in download_list if os.path.exists(dl[1])]
            success = batch_process_large_images(input_files, model_name, scale, face_enhance)
        else:
            success = run_ultra_fast_realesrgan(model_name, scale, face_enhance)
        
        if not success:
            return {"error": "Processing failed", "status": "failed"}
        
        # Upload results
        upload_list = []
        for file_info in files_info:
            input_filename = os.path.basename(file_info['input_file'])
            name, ext = os.path.splitext(input_filename)
            processed_filename = f"{name}_out{ext}"
            local_result_path = os.path.join(RESULT_FOLDER, processed_filename)
            
            if os.path.exists(local_result_path):
                upload_list.append((local_result_path, file_info['output_file']))
        
        upload_results = r2_handler.upload_files_parallel(upload_list)
        
        # Cleanup
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        shutil.rmtree(RESULT_FOLDER, ignore_errors=True)
        
        if use_ultra_mode and AGGRESSIVE_CLEANUP:
            GPUOptimizer.cleanup_gpu_memory()
        
        successful_uploads = sum(upload_results)
        gpu_info = GPUOptimizer.get_gpu_memory_info()
        
        # In the final result dictionary, ensure processed_files contains the R2 paths
        result = {
            "status": "completed",
            "processed_files": [upload_list[i][1] for i, success in enumerate(upload_results) if success],
            "model_used": model_name,
            "scale": scale,
            "total_files": successful_uploads,
            "ultra_mode": use_ultra_mode,
            "gpu_memory_used_gb": gpu_info['reserved_gb'] if gpu_info else 0,
            "performance_tier": "16GB_GPU_OPTIMIZED",
            "timestamp": datetime.now().isoformat()  # Add this line
        }
                
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        if AGGRESSIVE_CLEANUP:
            GPUOptimizer.cleanup_gpu_memory()
        return {"error": str(e), "status": "failed"}

def create_gradio_interface():
    """Create optimized Gradio interface"""
    interface = gr.Interface(
        fn=process_images,
        inputs=[
            gr.Textbox(label="API Key", type="password"),
            gr.Textbox(
                label="Files Data (JSON)", 
                placeholder='[{"input_file": "input1.jpg", "output_file": "output1.jpg"}]',
                lines=3
            ),
            gr.Dropdown(
                choices=[
                    "RealESRGAN_x4plus",
                    "RealESRGAN_x4plus_anime_6B", 
                    "RealESRGAN_x2plus",
                    "realesr-animevideov3",
                    "realesr-general-x4v3"
                ],
                value="RealESRGAN_x4plus",
                label="Model"
            ),
            gr.Slider(minimum=1, maximum=8, step=0.5, value=4.0, label="Scale"),
            gr.Checkbox(value=True, label="Face Enhancement"),
            gr.Checkbox(value=True, label="Ultra Mode (16GB GPU)")
        ],
        outputs=gr.JSON(label="Result"),
        title=" ULTRA-FAST Real-ESRGAN API",
        description="Optimized for 16GB GPU systems Maximum performance mode",
        allow_flagging="never"
    )
    return interface

def main():
    """Main server initialization"""
    logger.info("?? Starting ULTRA-FAST Real-ESRGAN Server")
    logger.info(f"System RAM: {SystemMonitor.get_available_ram_gb():.1f}GB")
    logger.info(f"CPU cores: {multiprocessing.cpu_count()}")
    
    # GPU initialization
    gpu_optimized = GPUOptimizer.optimize_cuda()
    if gpu_optimized:
        gpu_info = GPUOptimizer.get_gpu_memory_info()
        logger.info(f"GPU: {torch.cuda.get_device_name()} ({gpu_info['total_gb']:.1f}GB)")
        logger.info(f"VRAM usage limit: {GPU_MEMORY_FRACTION*100}% ({gpu_info['total_gb']*GPU_MEMORY_FRACTION:.1f}GB)")
    
    # Validate environment
    required_vars = ['R2_ACCESS_KEY', 'R2_SECRET_KEY', 'R2_ENDPOINT', 'R2_BUCKET']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        sys.exit(1)
    
    setup_folders()
    reset_shutdown_timer()
    
    # Launch interface
    interface = create_gradio_interface()
    logger.info("?? Launching Gradio interface...")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=True,
        enable_queue=True,
        max_threads=NUM_WORKERS
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
