"""
Apple Silicon GPU Acceleration Module
Provides Metal Performance Shaders (MPS) acceleration for M1/M2/M3/M4 chips
"""

import torch
import platform
import subprocess
import logging
import numpy as np

logger = logging.getLogger(__name__)

class AppleSiliconGPU:
    """GPU acceleration for Apple Silicon using Metal Performance Shaders"""
    
    def __init__(self):
        self.device = 'cpu'
        self.use_mps = False
        self.chip_model = None
        self.gpu_cores = None
        self.memory_gb = None
        self.batch_size = 4  # Default
        
        self._detect_hardware()
        self._initialize_mps()
        self._set_optimal_batch_size()
    
    def _detect_hardware(self):
        """Detect Apple Silicon chip model and hardware specs"""
        try:
            # Check if running on ARM Mac
            if platform.processor() != 'arm' or platform.system() != 'Darwin':
                logger.info("Not running on Apple Silicon")
                return
            
            # Get chip model
            result = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'])
            chip_name = result.decode().strip()
            
            # Parse chip model
            if 'M1' in chip_name:
                if 'Ultra' in chip_name:
                    self.chip_model = 'M1 Ultra'
                    self.gpu_cores = 64
                elif 'Max' in chip_name:
                    self.chip_model = 'M1 Max'
                    self.gpu_cores = 32
                elif 'Pro' in chip_name:
                    self.chip_model = 'M1 Pro'
                    self.gpu_cores = 16
                else:
                    self.chip_model = 'M1'
                    self.gpu_cores = 8
            elif 'M2' in chip_name:
                if 'Ultra' in chip_name:
                    self.chip_model = 'M2 Ultra'
                    self.gpu_cores = 76
                elif 'Max' in chip_name:
                    self.chip_model = 'M2 Max'
                    self.gpu_cores = 38
                elif 'Pro' in chip_name:
                    self.chip_model = 'M2 Pro'
                    self.gpu_cores = 19
                else:
                    self.chip_model = 'M2'
                    self.gpu_cores = 10
            elif 'M3' in chip_name:
                if 'Max' in chip_name:
                    self.chip_model = 'M3 Max'
                    self.gpu_cores = 40
                elif 'Pro' in chip_name:
                    self.chip_model = 'M3 Pro'
                    self.gpu_cores = 18
                else:
                    self.chip_model = 'M3'
                    self.gpu_cores = 10
            elif 'M4' in chip_name:
                if 'Max' in chip_name:
                    self.chip_model = 'M4 Max'
                    self.gpu_cores = 40
                elif 'Pro' in chip_name:
                    self.chip_model = 'M4 Pro'
                    self.gpu_cores = 20
                else:
                    self.chip_model = 'M4'
                    self.gpu_cores = 10
            else:
                self.chip_model = chip_name
                self.gpu_cores = 8  # Default
            
            # Get total memory
            result = subprocess.check_output(['sysctl', '-n', 'hw.memsize'])
            self.memory_gb = int(result.decode().strip()) / (1024 ** 3)
            
            logger.info(f"Detected Apple Silicon: {self.chip_model}")
            logger.info(f"GPU cores: {self.gpu_cores}, Memory: {self.memory_gb:.1f} GB")
            
        except Exception as e:
            logger.debug(f"Hardware detection failed: {e}")
    
    def _initialize_mps(self):
        """Initialize Metal Performance Shaders backend"""
        if not self.chip_model:
            return
        
        try:
            # Check PyTorch version
            torch_version = torch.__version__.split('+')[0]
            major, minor = map(int, torch_version.split('.')[:2])
            
            if major < 1 or (major == 1 and minor < 12):
                logger.warning(f"PyTorch {torch_version} too old for MPS (need 1.12+)")
                return
            
            # Check macOS version
            macos_version = platform.mac_ver()[0]
            macos_major, macos_minor = map(int, macos_version.split('.')[:2])
            
            if macos_major < 12 or (macos_major == 12 and macos_minor < 3):
                logger.warning(f"macOS {macos_version} too old for MPS (need 12.3+)")
                return
            
            # Test MPS availability
            if not torch.backends.mps.is_available():
                logger.warning("MPS backend not available")
                return
            
            # Create test tensor to verify MPS works
            test_tensor = torch.tensor([1.0, 2.0, 3.0])
            test_mps = test_tensor.to('mps')
            test_result = (test_mps * 2).cpu()
            
            if torch.allclose(test_result, torch.tensor([2.0, 4.0, 6.0])):
                self.device = torch.device('mps')
                self.use_mps = True
                logger.info("✓ Metal Performance Shaders (MPS) initialized successfully")
            else:
                logger.warning("MPS test failed")
                
        except Exception as e:
            logger.warning(f"MPS initialization failed: {e}")
    
    def _set_optimal_batch_size(self):
        """Set optimal batch size based on available memory"""
        if not self.use_mps or not self.memory_gb:
            return
        
        # Conservative batch sizing to avoid OOM
        if self.memory_gb < 8:
            self.batch_size = 2
        elif self.memory_gb < 16:
            self.batch_size = 4
        elif self.memory_gb < 32:
            self.batch_size = 8
        elif self.memory_gb < 64:
            self.batch_size = 12
        else:
            self.batch_size = 16
        
        logger.info(f"Optimal batch size: {self.batch_size}")
    
    def to_device(self, array):
        """
        Convert NumPy array to PyTorch tensor on MPS device
        
        Args:
            array: NumPy array
            
        Returns:
            PyTorch tensor on MPS device (or CPU if MPS unavailable)
        """
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        
        tensor = torch.from_numpy(array).float()
        
        if self.use_mps:
            return tensor.to(self.device)
        return tensor
    
    def to_numpy(self, tensor):
        """
        Convert PyTorch tensor back to NumPy array
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            NumPy array
        """
        if tensor.device.type == 'mps':
            tensor = tensor.cpu()
        return tensor.numpy()
    
    def empty_cache(self):
        """Clear MPS cache to free memory"""
        if self.use_mps:
            try:
                torch.mps.empty_cache()
                logger.debug("MPS cache cleared")
            except Exception as e:
                logger.debug(f"Failed to clear MPS cache: {e}")
    
    def get_memory_info(self):
        """
        Get current memory usage information
        
        Returns:
            Dict with memory statistics
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
            
            return {
                'total': mem.total / (1024 ** 3),  # GB
                'available': mem.available / (1024 ** 3),
                'used': mem.used / (1024 ** 3),
                'usage_percent': mem.percent
            }
        except ImportError:
            logger.warning("psutil not installed, cannot get memory info")
            return {
                'total': self.memory_gb if self.memory_gb else 0,
                'available': 0,
                'used': 0,
                'usage_percent': 0
            }
    
    def __repr__(self):
        return (f"AppleSiliconGPU(chip={self.chip_model}, "
                f"mps_enabled={self.use_mps}, "
                f"batch_size={self.batch_size})")
