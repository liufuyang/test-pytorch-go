
# How to setup Gotch

* Using python venv and a pip pytorch installed torchlib, as suggested [here](https://github.com/pytorch/pytorch/issues/13130#issuecomment-1402453615) and see [here](https://pypi.org/project/torch/1.11.0/#files)
* Plus a few fix around here and there (see below)

```
# using a venv at /some_path/.venv, use those two commands to verify before starting

which python
which pip

# the output above should look like this
# /some_path/.venv/bin/pip

# then install pytorch for M1 with ARM build.

pip install torch==1.11.0
```

Then setup environment parameters as usual, pointing the libtorch to the installed pytorch location in venv folder.
```
export GOTCH_LIBTORCH="/some_path/.venv/lib/python3.10/site-packages/torch/"
export LIBRARY_PATH="$LIBRARY_PATH:$GOTCH_LIBTORCH/lib"
export CPATH="$CPATH:$GOTCH_LIBTORCH/lib:$GOTCH_LIBTORCH/include:$GOTCH_LIBTORCH/include/torch/csrc/api/include"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$GOTCH_LIBTORCH/lib"
```

Then setup gotch with a script.
```
chmod +x setup-gotch.sh
export CUDA_VER=cpu && export GOTCH_VER=v0.7.0 && bash setup-gotch.sh

# If seeing the build faling of trying to load -lcuda, then try clean and setup again.
# go clean
# go clean -cache
```

You might also need to change 3 places on tensor.go. See
https://github.com/sugarme/gotch/issues/44#issuecomment-872557112

Now the build and run script should work as below.
```
go build . 
go run -exec "env DYLD_LIBRARY_PATH=$GOTCH_LIBTORCH/lib" .
```
