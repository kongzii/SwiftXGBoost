tmp_dir=$(mktemp -d -t xgboost-XXXXXX)
echo "Will work in $tmp_dir."

cd $tmp_dir
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build
cd build
cmake ..
make
make install
ldconfig

echo "Success!"
