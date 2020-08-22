tmp_dir=$(mktemp -d -t xgboost-XXXXXX)
echo "Will work in $tmp_dir."

cd $tmp_dir
git clone https://github.com/dmlc/xgboost
cd xgboost
git checkout tags/v1.1.1
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make
make install
ldconfig

echo "Success!"
