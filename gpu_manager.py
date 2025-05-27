"""
GPU Manager Module for Safe GPU Usage in Shared Environment
Designed for managing GPU allocation on a 4-GPU server
"""

import os
import torch
import subprocess
import psutil
from typing import Dict, List, Optional, Union



class GPUManager:
    """
    A class to manage GPU allocation and monitoring in a shared server environment.
    Ensures that notebooks use only specified GPUs or CPU to prevent conflicts.
    """

    def __init__(self, num_gpus: int = 4):
        """
        Initialize the GPU Manager.

        Parameters:
        -----------
        num_gpus : int, default=4
            The total number of GPUs available on the server.
            Default is 4 as specified for our server.
        """
        # Store the total number of GPUs available
        self.num_gpus = num_gpus

        # Track the current device setting
        # Possible values: None, "cpu", 0, 1, 2, 3 (GPU indices)
        self.current_setting: Optional[Union[str, int]] = None

        # Store the original CUDA_VISIBLE_DEVICES value when module is loaded
        # This helps us restore the original state if needed
        self.original_cuda_visible_devices: Optional[str] = os.environ.get('CUDA_VISIBLE_DEVICES')

        # Check if CUDA is available at initialization
        self.cuda_available: bool = torch.cuda.is_available()

        # If CUDA is available, store the actual device count that PyTorch sees
        if self.cuda_available:
            self.torch_visible_gpus: int = torch.cuda.device_count()
        else:
            self.torch_visible_gpus: int = 0

    def show_gpu_status(self) -> str:
        """
        Show status of all GPUs on the server.

        Returns a formatted string displaying:
            - GPU ID (0-3)
            - GPU Name
            - Memory usage (used/total in MB)
            - GPU utilization percentage
            - Running processes with PID and username

        Returns:
        --------
            str
            Formatted string with GPU status information

        Recommended Usage:
        ------------------
            Call this at the beginning of your notebook to:
            1. Check which GPUs are available before selecting one
            2. Verify no one is using the GPU you plan to use
            3. Monitor GPU usage during training (call periodically)

        Example:
        --------
        >> print(gpu_manager.show_gpu_status())
        """
        try:
            # Run nvidia-smi command to get GPU information
            # --query-gpu: specify which GPU properties to query
            # --format=csv,noheader,nounits: output format specification
            gpu_query_cmd = [
                'nvidia-smi',
                '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ]

            # Execute the command and capture output
            result = subprocess.run(
                gpu_query_cmd,
                capture_output=True,  # Capture stdout and stderr
                text=True,  # Return output as string (not bytes)
                check=True  # Raise exception if command fails
            )

            # Parse the GPU information
            gpu_info_lines = result.stdout.strip().split('\n')

            # Build the status report
            status_report = "=" * 80 + "\n"
            status_report += "GPU STATUS REPORT\n"
            status_report += "=" * 80 + "\n\n"

            # Process each GPU
            for gpu_line in gpu_info_lines:
                # Split CSV values: index, name, mem_used, mem_total, utilization
                parts = [p.strip() for p in gpu_line.split(',')]
                if len(parts) >= 5:
                    gpu_id = parts[0]
                    gpu_name = parts[1]
                    mem_used = parts[2]
                    mem_total = parts[3]
                    utilization = parts[4]

                    status_report += f"GPU {gpu_id}: {gpu_name}\n"
                    status_report += f"  Memory: {mem_used}MB / {mem_total}MB "
                    status_report += f"({float(mem_used) / float(mem_total) * 100:.1f}% used)\n"
                    status_report += f"  Utilization: {utilization}%\n"

                    # Get processes running on this GPU
                    processes = self._get_gpu_processes(int(gpu_id))
                    if processes:
                        status_report += "  Processes:\n"
                        for proc in processes:
                            status_report += f"    - PID: {proc['pid']}, "
                            status_report += f"User: {proc['user']}, "
                            status_report += f"Memory: {proc['gpu_memory']}MB\n"
                    else:
                        status_report += "  No processes running\n"

                    status_report += "\n"

            status_report += "=" * 80
            return status_report

        except subprocess.CalledProcessError as e:
            # nvidia-smi command failed
            return f"Error: Failed to query GPU status. nvidia-smi error: {e}"
        except FileNotFoundError:
            # nvidia-smi not found
            return "Error: nvidia-smi not found. Are NVIDIA drivers installed?"
        except Exception as e:
            # Other unexpected errors
            return f"Error: Unexpected error occurred: {type(e).__name__}: {e}"

    def _get_gpu_processes(self, gpu_id: int) -> List[Dict[str, Union[int, str]]]:
        """
        Get list of processes running on a specific GPU.

        This is a helper method (private, indicated by _ prefix).

        Parameters:
        -----------
        gpu_id : int
            The GPU index (0-3)

        Returns:
        --------
        List[Dict[str, Union[int, str]]]
            List of dictionaries, each containing:
            - 'pid': Process ID (int)
            - 'user': Username (str)
            - 'gpu_memory': GPU memory used in MB (int)
        """
        try:
            # Query processes on specific GPU
            # pmon stands for "process monitor"
            cmd = [
                'nvidia-smi',
                'pmon',  # Process monitoring mode
                '-c', '1',  # Count: sample once
                '-i', str(gpu_id)  # Specific GPU index
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            processes = []
            lines = result.stdout.strip().split('\n')

            # Skip header lines (first two lines are headers)
            for line in lines[2:]:
                parts = line.split()
                if len(parts) >= 6 and parts[1] != '-':
                    pid = int(parts[1])
                    # gpu_mem is in column 4 (0-indexed)
                    gpu_mem = int(parts[4]) if parts[4] != '-' else 0

                    # Get username for this PID using psutil
                    try:
                        process = psutil.Process(pid)
                        username = process.username()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        username = "unknown"

                    processes.append({
                        'pid': pid,
                        'user': username,
                        'gpu_memory': gpu_mem
                    })

            return processes

        except subprocess.CalledProcessError:
            # If pmon fails, try alternative method
            return self._get_gpu_processes_alternative(gpu_id)
        except Exception:
            return []

    def _get_gpu_processes_alternative(self, gpu_id: int) -> List[Dict[str, Union[int, str]]]:
        """
        Alternative method to get GPU processes using nvidia-smi query.

        Used as fallback if pmon method fails.
        """
        try:
            cmd = [
                'nvidia-smi',
                '--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits',
                f'--id={gpu_id}'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            processes = []
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        pid = int(parts[0].strip())
                        gpu_mem = int(parts[1].strip())

                        try:
                            process = psutil.Process(pid)
                            username = process.username()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            username = "unknown"

                        processes.append({
                            'pid': pid,
                            'user': username,
                            'gpu_memory': gpu_mem
                        })

            return processes
        except Exception:
            return []


    def get_gpu_info(self, gpu_id: int) -> Dict:
        """
        Get detailed information about a specific GPU.

        Provides comprehensive information about a single GPU including:
        - Basic info (name, driver version)
        - Memory details (used, free, total)
        - Temperature and power consumption
        - Clock speeds
        - Process list with detailed memory usage
        - Performance state

        Parameters:
        -----------
        gpu_id : int
            The GPU index (0-3) to query

        Returns:
        --------
        Dict
            Dictionary containing:
            - 'gpu_id': int - The GPU index
            - 'name': str - GPU model name
            - 'driver_version': str - NVIDIA driver version
            - 'memory': dict with 'used', 'free', 'total' in MB
            - 'temperature': int - GPU temperature in Celsius
            - 'power_draw': float - Current power draw in Watts
            - 'power_limit': float - Power limit in Watts
            - 'utilization': dict with 'gpu' and 'memory' percentages
            - 'clocks': dict with 'graphics' and 'memory' in MHz
            - 'performance_state': str - Current P-state (P0-P12)
            - 'processes': list of dicts with process information
            - 'error': str - Error message if query fails (optional)

        Raises:
        -------
        ValueError
            If gpu_id is not in valid range (0 to num_gpus-1)

        Recommended Usage:
        ------------------
        Call this when you need detailed info about a specific GPU:
        1. Before training: Check temperature, power state, and ensure GPU is healthy
        2. During debugging: Get detailed memory breakdown and clock speeds
        3. After selecting a GPU: Verify it has enough free memory for your model

        Example:
        --------
        > info = gpu_manager.get_gpu_info(2)
        > print(f"GPU 2 has {info['memory']['free']}MB free memory")
        > print(f"Temperature: {info['temperature']}°C")
        """
        # Validate GPU ID is in valid range
        if not 0 <= gpu_id < self.num_gpus:
            raise ValueError(
                f"Invalid gpu_id: {gpu_id}. "
                f"Must be between 0 and {self.num_gpus-1}"
            )

        # Initialize result dictionary with default values
        result = {
            'gpu_id': gpu_id,
            'name': 'Unknown',
            'driver_version': 'Unknown',
            'memory': {'used': 0, 'free': 0, 'total': 0},
            'temperature': 0,
            'power_draw': 0.0,
            'power_limit': 0.0,
            'utilization': {'gpu': 0, 'memory': 0},
            'clocks': {'graphics': 0, 'memory': 0},
            'performance_state': 'Unknown',
            'processes': []
        }

        try:
            # First, get driver version (global property, not per-GPU)
            driver_cmd = ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']
            driver_result = subprocess.run(
                driver_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            results = driver_result.stdout.strip().split("\n")
            result['driver_version'] = results[0]

            # Build comprehensive query for this specific GPU
            # We query many more properties than in show_gpu_status
            query_properties = [
                'name',                        # GPU model name
                'memory.used',                 # Used memory (MiB)
                'memory.free',                 # Free memory (MiB)
                'memory.total',                # Total memory (MiB)
                'utilization.gpu',             # GPU utilization (%)
                'utilization.memory',          # Memory controller utilization (%)
                'temperature.gpu',             # GPU temperature (C)
                'power.draw',                  # Current power draw (W)
                'power.limit',                 # Power limit (W)
                'clocks.current.graphics',     # Current graphics clock (MHz)
                'clocks.current.memory',       # Current memory clock (MHz)
                'pstate'                       # Performance state (P0-P12)
            ]

            # Construct the command
            gpu_cmd = [
                'nvidia-smi',
                f'--query-gpu={",".join(query_properties)}',
                '--format=csv,noheader,nounits',
                f'--id={gpu_id}'
            ]

            # Execute the query
            gpu_result = subprocess.run(
                gpu_cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse the CSV output
            values = [v.strip() for v in gpu_result.stdout.strip().split(',')]

            # Map values to result dictionary
            if len(values) >= len(query_properties):
                result['name'] = values[0]
                result['memory']['used'] = int(float(values[1]))
                result['memory']['free'] = int(float(values[2]))
                result['memory']['total'] = int(float(values[3]))
                result['utilization']['gpu'] = int(values[4])
                result['utilization']['memory'] = int(values[5])
                result['temperature'] = int(values[6])
                result['power_draw'] = float(values[7])
                result['power_limit'] = float(values[8])
                result['clocks']['graphics'] = int(values[9])
                result['clocks']['memory'] = int(values[10])
                result['performance_state'] = values[11]

            # Get detailed process information for this GPU
            result['processes'] = self._get_detailed_process_info(gpu_id)

            # Add computed convenience values
            result['memory']['used_percent'] = round(
                (result['memory']['used'] / result['memory']['total']) * 100, 1
            )
            result['power_draw_percent'] = round(
                (result['power_draw'] / result['power_limit']) * 100, 1
            ) if result['power_limit'] > 0 else 0

        except subprocess.CalledProcessError as e:
            result['error'] = f"Failed to query GPU {gpu_id}: {e.stderr if e.stderr else str(e)}"
        except Exception as e:
            result['error'] = f"Unexpected error querying GPU {gpu_id}: {str(e)}"

        return result

    def _get_detailed_process_info(self, gpu_id: int) -> List[Dict]:
        """
        Get detailed process information for a specific GPU.

        Enhanced version of _get_gpu_processes with more details.

        Parameters:
        -----------
        gpu_id : int
            The GPU index to query

        Returns:
        --------
        List[Dict]
            List of process dictionaries with enhanced information
        """
        processes = []

        try:
            # Use query-compute-apps for more detailed info
            cmd = [
                'nvidia-smi',
                '--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader,nounits',
                f'--id={gpu_id}'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        pid = int(parts[0])
                        process_name = parts[1]
                        gpu_memory = int(parts[2])

                        # Get additional process info using psutil
                        process_info = {
                            'pid': pid,
                            'name': process_name,
                            'gpu_memory_mb': gpu_memory,
                            'user': 'unknown',
                            'cpu_percent': 0.0,
                            'create_time': None,
                            'status': 'unknown'
                        }

                        try:
                            proc = psutil.Process(pid)
                            process_info['user'] = proc.username()
                            process_info['cpu_percent'] = proc.cpu_percent(interval=0.1)
                            process_info['create_time'] = proc.create_time()
                            process_info['status'] = proc.status()

                            # Calculate how long process has been running
                            import time
                            run_time = time.time() - process_info['create_time']
                            process_info['runtime_hours'] = round(run_time / 3600, 2)

                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                        processes.append(process_info)

            # Sort by GPU memory usage (descending)
            processes.sort(key=lambda x: x['gpu_memory_mb'], reverse=True)

        except Exception:
            # If detailed query fails, fall back to basic process list
            basic_processes = self._get_gpu_processes(gpu_id)
            for bp in basic_processes:
                processes.append({
                    'pid': bp['pid'],
                    'name': 'unknown',
                    'gpu_memory_mb': bp['gpu_memory'],
                    'user': bp['user'],
                    'cpu_percent': 0.0,
                    'create_time': None,
                    'status': 'unknown',
                    'runtime_hours': 0.0
                })

        return processes

    def use_gpu(self, gpu_id: int) -> Dict:
        """
        Configure the environment to use only the specified GPU.

        This function restricts the notebook to see and use ONLY the specified GPU.
        After calling this, the GPU will appear as 'cuda:0' to PyTorch, regardless
        of its physical ID.

        IMPORTANT: For reliable GPU isolation, call this BEFORE any PyTorch CUDA
        operations (including torch.cuda.is_available() or torch.cuda.device_count()).
        Once PyTorch initializes CUDA, changing CUDA_VISIBLE_DEVICES may not take
        effect properly.

        Parameters:
        -----------
        gpu_id : int
            The physical GPU ID (0-3) to use exclusively

        Returns:
        --------
        Dict
            Confirmation dictionary containing:
            - 'success': bool - Whether configuration was successful
            - 'physical_gpu_id': int - The actual GPU ID on the system
            - 'torch_gpu_id': int - How PyTorch will see it (always 0)
            - 'previous_setting': str - Previous CUDA_VISIBLE_DEVICES value
            - 'current_setting': str - New CUDA_VISIBLE_DEVICES value
            - 'torch_device': str - PyTorch device string ('cuda:0')
            - 'gpu_name': str - Name of the selected GPU
            - 'warnings': List[str] - Any warnings about current GPU state

        Raises:
        -------
        ValueError
            If gpu_id is not in valid range (0 to num_gpus-1)

        Recommended Usage:
        ------------------
        Call this at the START of your notebook, BEFORE:
        - Importing models or creating any PyTorch tensors
        - Calling gpu_manager.show_gpu_status() or get_current_status()
        - Any torch.cuda.* operations

        Best practice order:
        1. First cell: Import gpu_manager
        2. Second cell: gpu_manager.use_gpu(2) to select GPU 2
        3. Third cell: gpu_manager.show_gpu_status() to verify
        4. Fourth cell: Import transformers, load models, etc.

        Example:
        --------
        >> # VERY FIRST thing in notebook (before any CUDA operations)
        >> import gpu_manager
        >> result = gpu_manager.use_gpu(2)
        >>
        >> # NOW you can check status and load models
        >> print(gpu_manager.get_current_status())
        >> model = AutoModel.from_pretrained('dicta-il/dictabert')
        >> model.to('cuda:0')  # This will use physical GPU 2
        """
        # Validate GPU ID
        if not 0 <= gpu_id < self.num_gpus:
            raise ValueError(
                f"Invalid gpu_id: {gpu_id}. "
                f"Must be between 0 and {self.num_gpus - 1}"
            )

        # Initialize result dictionary
        result = {
            'success': False,
            'physical_gpu_id': gpu_id,
            'torch_gpu_id': 0,  # After setting, the GPU will appear as cuda:0
            'previous_setting': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
            'current_setting': str(gpu_id),
            'torch_device': 'cuda:0',
            'gpu_name': 'Unknown',
            'warnings': []
        }

        try:
            # Check if PyTorch CUDA has already been initialized
            cuda_already_initialized = False
            if hasattr(torch.cuda, '_initialized') and torch.cuda._initialized:
                cuda_already_initialized = True
            elif 'CUDA_VISIBLE_DEVICES' in os.environ and torch.cuda.device_count() > 1:
                # If we can see multiple devices despite CUDA_VISIBLE_DEVICES being set,
                # CUDA was likely initialized before
                cuda_already_initialized = True

            if cuda_already_initialized:
                result['warnings'].append(
                    "⚠️  WARNING: PyTorch CUDA appears to be already initialized. "
                    "Changing CUDA_VISIBLE_DEVICES now may not work properly. "
                    "For reliable GPU isolation, restart the kernel and call use_gpu() "
                    "BEFORE any PyTorch operations."
                )

            # Get current GPU info to check if it's in use
            gpu_info = self.get_gpu_info(gpu_id)
            result['gpu_name'] = gpu_info.get('name', 'Unknown')

            # Check for warnings about GPU state
            if gpu_info.get('memory', {}).get('used', 0) > 1000:  # > 1GB used
                result['warnings'].append(
                    f"GPU {gpu_id} already has {gpu_info['memory']['used']}MB in use. "
                    "Consider using a different GPU or checking existing processes."
                )

            if gpu_info.get('temperature', 0) > 80:
                result['warnings'].append(
                    f"GPU {gpu_id} is running hot ({gpu_info['temperature']}°C). "
                    "This may affect performance."
                )

            # Store the previous setting for potential restoration
            self.previous_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

            # Set CUDA_VISIBLE_DEVICES environment variable
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            # Update internal state
            self.current_setting = gpu_id

            # Verify PyTorch configuration
            if torch.cuda.is_available():
                # Force PyTorch to recognize the new setting
                # This is important if PyTorch was already initialized
                torch.cuda.empty_cache()

                # Get the actual device count PyTorch sees now
                device_count = torch.cuda.device_count()

                # Initialize torch_gpu_name variable
                torch_gpu_name = None

                if device_count == 1:
                    # Success - PyTorch sees exactly one GPU
                    result['success'] = True

                    # Get the name of the GPU as PyTorch sees it
                    try:
                        torch_gpu_name = torch.cuda.get_device_name(0)
                    except Exception:
                        torch_gpu_name = "Unknown GPU"

                    # Verify it matches what we expect
                    if result['gpu_name'] not in torch_gpu_name and torch_gpu_name not in result['gpu_name']:
                        result['warnings'].append(
                            f"GPU name mismatch. Expected '{result['gpu_name']}', "
                            f"but PyTorch reports '{torch_gpu_name}'"
                        )

                    else:
                        # Log the change for debugging
                        result['message'] = (
                            f"Successfully configured to use only GPU {gpu_id}. "
                            f"In PyTorch/HuggingFace, this GPU will appear as 'cuda:0'."
                        )

                elif device_count == 0:
                    result['warnings'].append(
                        "PyTorch cannot see any GPUs after setting. "
                        "This might be due to driver issues or CUDA not being available."
                    )
                else:
                    result['warnings'].append(
                        f"PyTorch sees {device_count} GPUs instead of 1. "
                        "This might cause unexpected behavior."
                    )
                    # Important warning about PyTorch initialization
                    result['warnings'].append(
                        "IMPORTANT: PyTorch CUDA was already initialized before setting GPU. "
                        "For reliable GPU isolation, set GPU BEFORE any PyTorch CUDA operations. "
                        "Consider restarting the kernel and calling use_gpu() first."
                    )

                    # Still try to get GPU name for diagnostic purposes
                    if device_count > 0:
                        try:
                            torch_gpu_name = torch.cuda.get_device_name(0)
                        except Exception:
                            torch_gpu_name = None

                    result['message'] = (
                        f"FAILED DUE TO NON-SAFE USAGE - "
                        f"Unsuccessfully configured to use only GPU {gpu_id}. "
                        f"Restart kernel before calling this function."
                    )

                # Add PyTorch-specific information
                result['torch_info'] = {
                    'cuda_available': True,
                    'device_count': device_count,
                    'current_device': torch.cuda.current_device() if device_count > 0 else None,
                    'device_name': torch_gpu_name  # Now safely defined
                    }


            else:
                result['warnings'].append(
                    "CUDA is not available to PyTorch. "
                    "GPU setting applied but PyTorch will use CPU."
                )
                result['torch_info'] = {
                    'cuda_available': False,
                    'device_count': 0,
                    'current_device': None,
                    'device_name': None
                }



        except Exception as e:
            result['success'] = False
            result['error'] = f"Failed to configure GPU: {str(e)}"
            result['warnings'].append(
                "An error occurred. GPU settings may be in an inconsistent state."
            )

        return result

    def use_cpu_only(self) -> Dict:
        """
        Configure the environment to use only CPU (no GPU access).

        This function completely disables GPU access for the notebook. After calling
        this, PyTorch/HuggingFace will not see any GPUs and will run all computations
        on CPU. This is useful for debugging, testing, or when GPUs are unavailable.

        Returns:
        --------
        Dict
            Confirmation dictionary containing:
            - 'success': bool - Whether configuration was successful
            - 'previous_setting': str - Previous CUDA_VISIBLE_DEVICES value
            - 'current_setting': str - New value (empty string)
            - 'cuda_available': bool - Should be False after setting
            - 'device_count': int - Should be 0
            - 'device': str - Will be 'cpu'
            - 'warnings': List[str] - Any warnings
            - 'message': str - Confirmation message

        Recommended Usage:
        ------------------
        Call this at the START of your notebook, BEFORE importing models:
        1. When debugging to ensure reproducible CPU-only results
        2. When all GPUs are busy and you want to test on CPU
        3. When developing/testing code before GPU training
        4. When you need deterministic behavior (CPU is more deterministic)

        Example:
        --------
        >> # Force CPU-only mode
        >> result = gpu_manager.use_cpu_only()
        >> print(result['message'])
        >>
        >> # Now all models will use CPU
        >> model = AutoModel.from_pretrained('dicta-il/dictabert')
        >> print(model.device)  # Will show 'cpu'
        >>
        >> # This will automatically use CPU (no .cuda() needed)
        >> trainer = Trainer(model=model, ...)
        """
        # Initialize result dictionary
        result = {
            'success': False,
            'previous_setting': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
            'current_setting': '',  # Empty string means no GPUs visible
            'cuda_available': True,  # Will be updated
            'device_count': -1,  # Will be updated
            'device': 'cpu',
            'warnings': [],
            'message': ''
        }

        try:
            # Check current GPU usage before switching to CPU
            if torch.cuda.is_available():
                current_device_count = torch.cuda.device_count()

                # Warn if GPUs are currently accessible
                if current_device_count > 0:
                    result['warnings'].append(
                        f"Disabling access to {current_device_count} GPU(s). "
                        "All computations will now run on CPU."
                    )

                # Check if any tensors are currently on GPU
                # This is informational - we can't move existing tensors
                if torch.cuda.memory_allocated() > 0:
                    allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    result['warnings'].append(
                        f"Warning: {allocated_mb:.1f}MB is currently allocated on GPU. "
                        "Existing GPU tensors will remain on GPU, but new tensors will be on CPU."
                    )

            # Store the previous setting for potential restoration
            self.previous_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

            # Set CUDA_VISIBLE_DEVICES to empty string
            # Empty string means CUDA sees no devices (different from not set!)
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

            # Update internal state
            self.current_setting = 'cpu'

            # Force PyTorch to recognize the change
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Verify the setting worked
            cuda_available_after = torch.cuda.is_available()
            device_count_after = torch.cuda.device_count() if cuda_available_after else 0

            # Check if setting was successful
            if not cuda_available_after or device_count_after == 0:
                result['success'] = True
                result['cuda_available'] = False
                result['device_count'] = 0
                result['message'] = (
                    "Successfully configured CPU-only mode. "
                    "No GPUs are visible to PyTorch/HuggingFace. "
                    "All computations will run on CPU."
                )

                # Add performance warning for large models
                result['warnings'].append(
                    "Note: CPU training will be significantly slower than GPU. "
                    "Consider using a smaller batch size to avoid out-of-memory errors."
                )
            else:
                # Setting failed somehow
                result['warnings'].append(
                    f"Failed to completely disable GPU access. "
                    f"PyTorch still sees {device_count_after} GPU(s). "
                    "Try restarting the kernel and setting CPU-only mode first."
                )
                result['cuda_available'] = cuda_available_after
                result['device_count'] = device_count_after

            # Add CPU information
            result['cpu_info'] = self._get_cpu_info()

            # Add memory warnings based on available RAM
            if result['cpu_info']['available_memory_gb'] < 16:
                result['warnings'].append(
                    f"Warning: Only {result['cpu_info']['available_memory_gb']:.1f}GB RAM available. "
                    "Consider using a smaller model or batch size."
                )

        except Exception as e:
            result['success'] = False
            result['error'] = f"Failed to configure CPU-only mode: {str(e)}"
            result['warnings'].append(
                "An error occurred. GPU settings may be in an inconsistent state."
            )

        return result

    def _get_cpu_info(self) -> Dict[str, Union[int, float, str]]:
        """
        Get CPU and memory information for the system.

        Returns:
        --------
        Dict containing CPU and memory statistics
        """
        cpu_info = {
            'cpu_count': os.cpu_count() or 0, # Amount of logical cpus
            'cpu_percent': 0.0,
            'total_memory_gb': 0.0,
            'available_memory_gb': 0.0,
            'used_memory_gb': 0.0,
            'memory_percent': 0.0
        }

        try:
            # Get CPU usage percentage
            cpu_info['cpu_percent'] = psutil.cpu_percent(interval=0.1)

            # Get memory information
            memory = psutil.virtual_memory()
            cpu_info['total_memory_gb'] = memory.total / (1024 ** 3)  # Convert to GB
            cpu_info['available_memory_gb'] = memory.available / (1024 ** 3)
            cpu_info['used_memory_gb'] = memory.used / (1024 ** 3)
            cpu_info['memory_percent'] = memory.percent

        except Exception:
            # If psutil fails, at least we have the defaults
            pass

        return cpu_info


    def verify_trainer_device(self, trainer) -> Dict:
        """
        Verify that a HuggingFace Trainer will use the correct device.

        Examines a Trainer object's configuration and confirms it matches the
        current GPU/CPU settings of this module. Provides "proof" that the
        trainer is correctly configured.

        Parameters:
        -----------
        trainer : transformers.Trainer
            A HuggingFace Trainer instance to verify

        Returns:
        --------
        Dict
            Verification results containing:
            - 'matches_module_setting': bool - Whether trainer matches module config
            - 'module_setting': str/int/None - Current module setting
            - 'trainer_device': str - Device trainer will use (e.g., 'cuda:0', 'cpu')
            - 'trainer_n_gpu': int - Number of GPUs trainer thinks it has
            - 'trainer_no_cuda': bool - Whether trainer has CUDA disabled
            - 'physical_gpu_id': int/None - Actual GPU ID if using GPU
            - 'current_visible_devices': str/None - CUDA_VISIBLE_DEVICES value
            - 'warnings': List[str] - Any configuration warnings
            - 'recommendations': List[str] - Suggestions to fix issues
            - 'is_correct': bool - Simple yes/no: is configuration correct?

        Recommended Usage:
        ------------------
        ALWAYS call this after creating a Trainer but BEFORE calling trainer.train():
        1. Create your Trainer with TrainingArguments
        2. Call verify_trainer_device(trainer) to ensure correct GPU/CPU
        3. Only proceed with training if is_correct=True

        Example:
        --------
        >> # Set GPU first
        >> gpu_manager.use_gpu(2)
        >>
        >> # Create trainer
        >> training_args = TrainingArguments(output_dir="./results", ...)
        >> trainer = Trainer(model=model, args=training_args, ...)
        >>
        >> # Verify before training
        >> verification = gpu_manager.verify_trainer_device(trainer)
        >> if verification['is_correct']:
        >>     print(f"✓ Trainer will use {verification['trainer_device']}")
        >>     trainer.train()
        >> else:
        >>     print("✗ Configuration mismatch!")
        >>     for warning in verification['warnings']:
        >>         print(f"  - {warning}")
        """
        # Initialize result dictionary
        result = {
            'matches_module_setting': False,
            'module_setting': self.current_setting,
            'trainer_device': 'unknown',
            'trainer_n_gpu': 0,
            'trainer_no_cuda': False,
            'physical_gpu_id': None,
            'current_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'warnings': [],
            'recommendations': [],
            'is_correct': False
        }

        try:
            # Check if trainer has the expected attributes
            if not hasattr(trainer, 'args'):
                result['warnings'].append(
                    "Object does not appear to be a HuggingFace Trainer (no 'args' attribute)"
                )
                return result

            # Extract trainer configuration
            training_args = trainer.args

            # Get device information from trainer
            if hasattr(training_args, 'device'):
                device = training_args.device
                # Convert torch.device to string if needed
                result['trainer_device'] = str(device)
            else:
                result['warnings'].append("Trainer args missing 'device' attribute")
                return result

            # Get GPU count and CUDA settings
            result['trainer_n_gpu'] = getattr(training_args, 'n_gpu', 0)
            result['trainer_no_cuda'] = getattr(training_args, 'no_cuda', False)

            # Check if using distributed training
            is_distributed = (
                    hasattr(training_args, 'local_rank') and
                    training_args.local_rank != -1
            )
            if is_distributed:
                result['warnings'].append(
                    "Trainer is configured for distributed training. "
                    "GPU verification may not be accurate for multi-GPU setups."
                )

            # Determine what device the trainer will actually use
            trainer_will_use_cuda = (
                    result['trainer_device'].startswith('cuda') and
                    not result['trainer_no_cuda']
            )

            # Now verify against module settings
            if self.current_setting is None:
                # Module hasn't set any restrictions
                result['matches_module_setting'] = True
                result['warnings'].append(
                    "GPU module hasn't set any device restrictions. "
                    "Trainer will use its default device selection."
                )

            elif self.current_setting == 'cpu':
                # Module set CPU-only mode
                if trainer_will_use_cuda:
                    result['matches_module_setting'] = False
                    result['warnings'].append(
                        "Module is in CPU-only mode, but Trainer is configured to use CUDA!"
                    )
                    result['recommendations'].append(
                        "Either: 1) Set no_cuda=True in TrainingArguments, or "
                        "2) Call gpu_manager.use_cpu_only() before creating Trainer"
                    )
                else:
                    result['matches_module_setting'] = True
                    if result['trainer_device'] != 'cpu':
                        result['warnings'].append(
                            f"Unexpected device string: {result['trainer_device']}. "
                            "Expected 'cpu' for CPU-only mode."
                        )

            else:
                # Module set specific GPU
                expected_visible_devices = str(self.current_setting)
                actual_visible_devices = result['current_visible_devices']

                if actual_visible_devices != expected_visible_devices:
                    result['warnings'].append(
                        f"CUDA_VISIBLE_DEVICES mismatch! "
                        f"Module expects '{expected_visible_devices}', "
                        f"but environment has '{actual_visible_devices}'"
                    )
                    result['recommendations'].append(
                        "The environment variable may have been changed. "
                        "Try calling gpu_manager.use_gpu() again."
                    )

                if trainer_will_use_cuda:
                    # Trainer will use CUDA, check if it's the right device
                    if result['trainer_n_gpu'] == 1:
                        # Good - trainer sees exactly one GPU
                        result['matches_module_setting'] = True
                        result['physical_gpu_id'] = self.current_setting

                        # Extract CUDA device index from trainer device string
                        if ':' in result['trainer_device']:
                            trainer_cuda_idx = int(result['trainer_device'].split(':')[1])
                            if trainer_cuda_idx != 0:
                                result['warnings'].append(
                                    f"Trainer using cuda:{trainer_cuda_idx} instead of cuda:0. "
                                    "This is unusual when CUDA_VISIBLE_DEVICES is set."
                                )
                    elif result['trainer_n_gpu'] == 0:
                        result['warnings'].append(
                            "Trainer reports 0 GPUs but device is CUDA. "
                            "This indicates a configuration problem."
                        )
                        result['recommendations'].append(
                            "Check if CUDA is properly initialized. "
                            "Try creating the Trainer after setting GPU with gpu_manager."
                        )
                    else:
                        result['warnings'].append(
                            f"Trainer sees {result['trainer_n_gpu']} GPUs, "
                            f"but module restricted to GPU {self.current_setting} only."
                        )
                else:
                    # Module set GPU but trainer not using CUDA
                    result['matches_module_setting'] = False
                    result['warnings'].append(
                        f"Module configured GPU {self.current_setting}, "
                        f"but Trainer is not using CUDA!"
                    )
                    result['recommendations'].append(
                        "Check if no_cuda=True was set in TrainingArguments. "
                        "Remove no_cuda=True to use GPU."
                    )

            # Check for common issues
            if trainer_will_use_cuda and torch.cuda.device_count() == 0:
                result['warnings'].append(
                    "CRITICAL: Trainer expects CUDA but PyTorch sees 0 GPUs! "
                    "Training will fail."
                )
                result['recommendations'].append(
                    "Ensure gpu_manager.use_gpu() is called BEFORE creating Trainer."
                )

            # Set final verification result
            result['is_correct'] = (
                    result['matches_module_setting'] and
                    len([w for w in result['warnings'] if 'CRITICAL' in w]) == 0
            )

            # Add helpful summary message
            if result['is_correct']:
                if self.current_setting == 'cpu':
                    device_desc = "CPU only"
                elif self.current_setting is not None:
                    device_desc = f"GPU {self.current_setting} (appears as cuda:0)"
                else:
                    device_desc = f"{result['trainer_device']}"

                result['summary'] = f"✓ Trainer correctly configured to use {device_desc}"
            else:
                result['summary'] = (
                    "✗ Configuration mismatch detected. "
                    "See warnings and recommendations."
                )

        except Exception as e:
            result['warnings'].append(f"Error during verification: {str(e)}")
            result['recommendations'].append(
                "Ensure you're passing a valid HuggingFace Trainer object."
            )
            result['is_correct'] = False
            result['summary'] = "✗ Verification failed with error"

        return result

    def get_current_status(self):
        """
        Provide a readable visualization of the current environment status.

        Shows:
            - Current CUDA_VISIBLE_DEVICES setting
            - What PyTorch/HuggingFace can see
            - Module's current configuration
            - GPU/CPU utilization details

            Returns:
            --------
            str
            Formatted string with complete environment status

            Recommended Usage:
            ------------------
            Call this anytime to understand the current state:
            1. After setting GPU/CPU to confirm configuration
            2. Before creating models/trainers to verify environment
            3. When debugging device-related issues
            4. At the start of notebook to see initial state

            Example:
            --------
            >> print(gpu_manager.get_current_status())
            """
        # Build status report with nice formatting
        status = "\n" + "=" * 80 + "\n"
        status += "🖥️  GPU MANAGER ENVIRONMENT STATUS\n"
        status += "=" * 80 + "\n\n"

        # Section 1: Environment Variables
        status += "📋 ENVIRONMENT VARIABLES\n"
        status += "-" * 40 + "\n"
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_visible is None:
            status += "CUDA_VISIBLE_DEVICES: <not set>\n"
            status += "→ Interpretation: All GPUs are visible (default behavior)\n"
        elif cuda_visible == '':
            status += "CUDA_VISIBLE_DEVICES: '' (empty string)\n"
            status += "→ Interpretation: No GPUs visible (CPU-only mode)\n"
        else:
            status += f"CUDA_VISIBLE_DEVICES: '{cuda_visible}'\n"
            # Parse the value to show which GPUs
            gpu_list = [g.strip() for g in cuda_visible.split(',') if g.strip()]
            if len(gpu_list) == 1:
                status += f"→ Interpretation: Only GPU {gpu_list[0]} is visible\n"
            else:
                status += f"→ Interpretation: GPUs {', '.join(gpu_list)} are visible\n"
        status += "\n"

        # Section 2: PyTorch/CUDA Status
        status += "🔥 PYTORCH/CUDA STATUS\n"
        status += "-" * 40 + "\n"

        cuda_available = torch.cuda.is_available()
        status += f"CUDA Available: {cuda_available}\n"

        if cuda_available:
            device_count = torch.cuda.device_count()
            status += f"Visible GPU Count: {device_count}\n"

            if device_count > 0:
                status += "Visible Devices:\n"
                for i in range(device_count):
                    try:
                        device_name = torch.cuda.get_device_name(i)
                        # Get memory info for this device
                        torch.cuda.set_device(i)
                        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)

                        status += f"  • cuda:{i}: {device_name}\n"
                        status += f"    Memory: {allocated:.1f}GB allocated / "
                        status += f"{reserved:.1f}GB reserved / {total_mem:.1f}GB total\n"
                    except Exception as e:
                        status += f"  • cuda:{i}: <error accessing device>\n"
            else:
                status += "→ No GPUs visible to PyTorch\n"
        else:
            status += "→ CUDA is not available (CPU-only mode or no CUDA installation)\n"

        # Current PyTorch default device
        if cuda_available and torch.cuda.device_count() > 0:
            current_device = torch.cuda.current_device()
            status += f"\nCurrent Default CUDA Device: cuda:{current_device}\n"

        status += "\n"

        # Section 3: Module Configuration
        status += "⚙️  GPU MANAGER MODULE STATUS\n"
        status += "-" * 40 + "\n"

        if self.current_setting is None:
            status += "Module State: No configuration applied\n"
            status += "→ Module has not changed any GPU settings\n"
            status += "→ System is using default GPU visibility\n"
        elif self.current_setting == 'cpu':
            status += "Module State: CPU-ONLY MODE\n"
            status += "→ Module has disabled all GPU access\n"
            status += "→ All computations will run on CPU\n"

            # Add CPU info
            cpu_info = self._get_cpu_info()
            status += f"\nCPU Information:\n"
            status += f"  • CPU Cores: {cpu_info['cpu_count']}\n"
            status += f"  • CPU Usage: {cpu_info['cpu_percent']:.1f}%\n"
            status += f"  • RAM: {cpu_info['used_memory_gb']:.1f}GB used / "
            status += f"{cpu_info['total_memory_gb']:.1f}GB total "
            status += f"({cpu_info['memory_percent']:.1f}%)\n"
            status += f"  • Available RAM: {cpu_info['available_memory_gb']:.1f}GB\n"
        else:
            # Specific GPU selected
            status += f"Module State: GPU {self.current_setting} SELECTED\n"
            status += f"→ Module has restricted visibility to GPU {self.current_setting} only\n"
            status += f"→ This GPU appears as 'cuda:0' to PyTorch\n"

            # Get info about the selected GPU
            try:
                gpu_info = self.get_gpu_info(self.current_setting)
                if 'error' not in gpu_info:
                    status += f"\nSelected GPU Details:\n"
                    status += f"  • Physical ID: GPU {gpu_info['gpu_id']}\n"
                    status += f"  • Model: {gpu_info['name']}\n"
                    status += f"  • Memory: {gpu_info['memory']['used']}MB / "
                    status += f"{gpu_info['memory']['total']}MB "
                    status += f"({gpu_info['memory']['used_percent']:.1f}%)\n"
                    status += f"  • Temperature: {gpu_info['temperature']}°C\n"
                    status += f"  • Utilization: {gpu_info['utilization']['gpu']}%\n"

                    if gpu_info['processes']:
                        status += f"  • Running Processes: {len(gpu_info['processes'])}\n"
                        for proc in gpu_info['processes'][:3]:  # Show first 3
                            status += f"    - PID {proc['pid']}: {proc['user']} "
                            status += f"({proc['gpu_memory_mb']}MB)\n"
                        if len(gpu_info['processes']) > 3:
                            status += f"    ... and {len(gpu_info['processes']) - 3} more\n"
            except Exception:
                status += "  • <Unable to get GPU details>\n"

        status += "\n"

        # Section 4: Recommendations
        status += "💡 RECOMMENDATIONS\n"
        status += "-" * 40 + "\n"

        if self.current_setting is None:
            status += "• No GPU restrictions active\n"
            status += "• Call use_gpu(N) to restrict to specific GPU\n"
            status += "• Call use_cpu_only() to force CPU usage\n"
        elif self.current_setting == 'cpu':
            status += "• CPU-only mode is active\n"
            status += "• Remember to set no_cuda=True in TrainingArguments\n"
            status += "• Use smaller batch sizes to avoid OOM\n"
            if cuda_available:
                status += "• Call use_gpu(N) to switch back to GPU\n"
        else:
            # GPU selected
            if cuda_available and torch.cuda.device_count() == 1:
                status += "✓ Environment correctly configured for single GPU usage\n"
                status += "• Create Trainer objects now - they will use this GPU\n"
            else:
                status += "⚠️  Mismatch between module setting and PyTorch visibility\n"
                status += "• Try calling use_gpu() again or restart kernel\n"

        # Section 5: Quick Commands
        status += "\n📝 QUICK COMMANDS\n"
        status += "-" * 40 + "\n"
        status += "• See all GPUs: gpu_manager.show_gpu_status()\n"
        status += "• Use specific GPU: gpu_manager.use_gpu(N)\n"
        status += "• Use CPU only: gpu_manager.use_cpu_only()\n"
        status += "• Verify trainer: gpu_manager.verify_trainer_device(trainer)\n"

        status += "\n" + "=" * 80 + "\n"

        return status

# Create a singleton instance for easy module-level access
_gpu_manager_instance = GPUManager()

# Expose methods at module level for convenience
show_gpu_status = _gpu_manager_instance.show_gpu_status
get_gpu_info = _gpu_manager_instance.get_gpu_info
use_gpu = _gpu_manager_instance.use_gpu
use_cpu_only = _gpu_manager_instance.use_cpu_only
verify_trainer_device = _gpu_manager_instance.verify_trainer_device
get_current_status = _gpu_manager_instance.get_current_status