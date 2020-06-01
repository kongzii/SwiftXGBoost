tmp_dir=$(mktemp -d -t xgboost-XXXXXX)
echo "Will work in $tmp_dir."

cd $tmp_dir
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
git checkout release_1.1.0
mkdir build
cd build
cmake -DCMAKE_INSTALL_LIBDIR=/usr/lib ..
make
make install
ldconfig

echo "Success!"
