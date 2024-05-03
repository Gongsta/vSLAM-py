
### Older
`main_zed.py` was my attempt at not using any form of abstractions. Just have all of it there. It's not very readable. I also tried using VPI,
but it's very annoying to use, and there are discrepancies with OpenCV implementation. Could not replicate results.

Installation
```
VPI for hardware-accelerated image processing
```bash
sudo apt install python3.8-vpi2
```


You might have VPI3 if you are on the latest. In which case run
```bash
sudo apt install python3.9-vpi3
```

#### Setting up the zed
```
cd /usr/local/zed/
python3 get_python_api.py
```

### Design Decisions
Initially, I was desigining the entire codebase in [C++](https://github.com/Gongsta/vSLAM). For
hardware accceleration, I used the help of the VPI library. I got the entire way there,
until I had to do optimization. This was a pain to do in C++ due to lack of support. I decided
to transfer everything to Python. VPI has a Python API, but I saw that OpenCV with CUDA had
all the bindings anyways.
