# GPU Manager – README
Python module providing gpu management API for Reichman's NLP lab
A single-file utility for **safe, conflict-free GPU usage on multi-user, 4-GPU servers**.
It wraps common GPU/CPU-selection tasks, status reporting and sanity checks into a
notebook-friendly API.

## Quick start

```python
import gpu_manager   # this creates a singleton instance for you

# Show who is using which GPU right now
print(gpu_manager.show_gpu_status())

# Grab GPU 2 exclusively BEFORE any torch code touches CUDA
gpu_manager.use_gpu(2)

# …train as usual – PyTorch sees only one device, cuda:0…
```

---

## Public API & Examples

Below you’ll find *every* example that appears in the code’s doc-strings, so you
can copy-paste them straight into your notebook or script.

| Function                                     | Purpose                                                                                |
| -------------------------------------------- | -------------------------------------------------------------------------------------- |
| `gpu_manager.show_gpu_status()`              | Text table with live utilisation, memory and processes for every GPU.                  |
| `gpu_manager.get_gpu_info(gpu_id)`           | Detailed stats for one GPU (temp, power, clocks, processes…).                          |
| `gpu_manager.use_gpu(gpu_id)`                | Locks the environment to **one** physical GPU (appears as `cuda:0`).                   |
| `gpu_manager.use_cpu_only()`                 | Hides all GPUs – forces full CPU execution for debugging / fallback.                   |
| `gpu_manager.verify_trainer_device(trainer)` | Confirms a Hugging Face `Trainer` is really going to the intended device.              |
| `gpu_manager.get_current_status()`           | Multi-section report of **everything** (env vars, PyTorch view, module state). |

### 1 ️ `show_gpu_status`

```python
>> print(gpu_manager.show_gpu_status())
```

---

### 2 ️`get_gpu_info`

```python
> info = gpu_manager.get_gpu_info(2)
> print(f"GPU 2 has {info['memory']['free']} MB free memory")
> print(f"Temperature: {info['temperature']} °C")
```

---

### 3 ️`use_gpu`

```python
>> # VERY FIRST cell (before any CUDA ops)
>> import gpu_manager
>> result = gpu_manager.use_gpu(2)

>> # NOW you can check status and load models
>> print(gpu_manager.get_current_status())
>> model = AutoModel.from_pretrained('dicta-il/dictabert')
```

---

### 4 ️`use_cpu_only`

```python
>> # Force CPU-only mode
>> result = gpu_manager.use_cpu_only()
>> print(result['message'])

>> # All models will now run on CPU
>> model = AutoModel.from_pretrained('dicta-il/dictabert')
>> print(model.device)          # → 'cpu'

>> # No .cuda() call needed
>> trainer = Trainer(model=model, ...)
```

---

### 5 ️`verify_trainer_device`

```python
>> # Set GPU first
>> gpu_manager.use_gpu(2)

>> # Create trainer
>> training_args = TrainingArguments(output_dir="./results", ...)
>> trainer = Trainer(model=model, args=training_args, ...)

>> # Verify before training
>> verification = gpu_manager.verify_trainer_device(trainer)
>> if verification['is_correct']:
>>     print(f"✓ Trainer will use {verification['trainer_device']}")
>>     trainer.train()
>> else:
>>     print("✗ Configuration mismatch!")
>>     for warning in verification['warnings']:
>>         print(f"  - {warning}")
```

---

### 6 ️`get_current_status`

```python
>> print(gpu_manager.get_current_status())
```

---

## Best-practice workflow

1. **Immediately after kernel start**

   ```python
   import gpu_manager
   gpu_manager.use_gpu(1)          # or gpu_manager.use_cpu_only()
   ```

2. **Confirm**

   ```python
   print(gpu_manager.get_current_status())
   ```

3. **Load models / data and build your `Trainer`**.

4. **Double-check**

   ```python
   gpu_manager.verify_trainer_device(trainer)
   ```

---

## Notes & Caveats

* **Call `use_gpu()` / `use_cpu_only()` before *any* CUDA interaction**, otherwise
  PyTorch has already initialised and the change might not stick.
* The helper prints warnings if the selected GPU is hot or already busy.
* Everything is also available as a **singleton module-level shortcut** –
  feel free to `from gpu_manager import use_gpu`.

---
