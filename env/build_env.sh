echo "prepare virtual env"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash  ~/miniconda.sh -b -p miniconda
export PATH="PATH:${pwd}/miniconda/bin:$PATH"
echo "install env libraries"
conda env create -n py36tftest -f ../py36tf.yml

echo "list what we have"
. activate py36tftest
conda list

echo "enviroment build successfully"
. deactivate
