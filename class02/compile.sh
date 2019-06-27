#i/binbash

nvcc -o run $1 -L/usr/local/cuda/lib64 -lcurand
echo "nvcc"
./run
