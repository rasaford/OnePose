# do not specify a shell here to keep the current conda environment active

# taken from DeelLM/example.sh (hidden build instructions arrgh!!)
cd DeepLM
git submodule update --init --recursive
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON
make -j
cd ../..

export TORCH_USE_RTLD_GLOBAL=YES

